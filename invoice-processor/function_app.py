import azure.functions as func
import logging
import json
import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp()

# ── Configuration from environment variables ────────────────────────
DI_ENDPOINT = os.environ["DI_ENDPOINT"]
DI_KEY = os.environ["DI_KEY"]
MODEL_ID = os.environ["MODEL_ID"]
STORAGE_CONN = os.environ.get("STORAGE_CONNECTION", "")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.90"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.90"))
FALLBACK_THRESHOLD = float(os.environ.get("FALLBACK_THRESHOLD", "0.40"))
PREBUILT_MODEL_ID = "prebuilt-invoice"

# ════════════════════════════════════════════════════════════════════
#  TRIGGER 1 — HTTP (used by Power Automate for email-driven flow)
# ════════════════════════════════════════════════════════════════════
@app.route(route="extract", auth_level=func.AuthLevel.FUNCTION)
def extract_invoice_http(req: func.HttpRequest) -> func.HttpResponse:
    """Receives a PDF as raw bytes, returns flattened JSON for Power Automate."""
    logging.info("HTTP trigger received an invoice.")

    try:
        pdf_bytes = req.get_body()
        if not pdf_bytes:
            return _json_response({"error": "No file content in request body"}, 400)

        extracted = run_extraction(pdf_bytes, "email_attachment")
        flat = flatten_for_power_automate(extracted)
        return _json_response(flat, 200)

    except Exception as e:
        logging.error(f"HTTP extraction error: {str(e)}")
        return _json_response({"error": str(e), "exception_status": "Error"}, 500)


# ════════════════════════════════════════════════════════════════════
#  TRIGGER 2 — BLOB (used for batch/drop-folder processing)
# ════════════════════════════════════════════════════════════════════
@app.blob_trigger(
    arg_name="myblob",
    path="invoices-input/{name}",
    connection="STORAGE_CONNECTION"
)
def process_invoice_blob(myblob: func.InputStream):
    """Triggered when a PDF lands in the invoices-input container."""
    logging.info(f"Blob trigger: {myblob.name}, {myblob.length} bytes")

    try:
        pdf_bytes = myblob.read()
        extracted = run_extraction(pdf_bytes, myblob.name)

        needs_review = extracted["exception_status"] == "Review"
        output_container = "invoices-review" if needs_review else "invoices-approved"
        write_result_to_blob(extracted, output_container, myblob.name)

        logging.info(f"Routed to {output_container} | status={extracted['exception_status']}")

    except Exception as e:
        logging.error(f"Blob extraction error for {myblob.name}: {str(e)}")
        write_error_to_blob(myblob.name, str(e))
        raise


# ════════════════════════════════════════════════════════════════════
#  SHARED CORE — extraction logic used by both triggers
# ════════════════════════════════════════════════════════════════════
# Prebuilt invoice model ID — built into Azure, no training needed
def run_extraction(pdf_bytes, source_name):
    client = DocumentIntelligenceClient(
        endpoint=DI_ENDPOINT,
        credential=AzureKeyCredential(DI_KEY)
    )

    # ── Step 1: Try custom model first ──────────────────────────────
    logging.info(f"Trying custom model: {MODEL_ID}")
    result = analyze_document(client, MODEL_ID, pdf_bytes)

    if result.documents:
        doc = result.documents[0]
        extracted = extract_fields(doc, source_name)
        min_conf = extracted["_min_confidence"]

        # ── Step 2: Use custom model UNLESS it clearly failed ───────
        #     Fall back ONLY if confidence is catastrophically low
        #     (likely a completely different vendor), NOT just because
        #     it's below the review threshold.
        if min_conf >= FALLBACK_THRESHOLD:
            logging.info(
                f"Custom model usable ({min_conf:.2f} >= {FALLBACK_THRESHOLD}) "
                f"→ using custom model result"
            )
            extracted["_model_used"] = MODEL_ID
            return finalize(extracted)

        logging.info(
            f"Custom model confidence catastrophically low "
            f"({min_conf:.2f} < {FALLBACK_THRESHOLD}) "
            f"→ falling back to '{PREBUILT_MODEL_ID}'"
        )
    else:
        logging.warning(
            f"Custom model returned no documents "
            f"→ falling back to '{PREBUILT_MODEL_ID}'"
        )

    # ── Step 3: Prebuilt-invoice as fallback ─────────────────────────
    result_prebuilt = analyze_document(client, PREBUILT_MODEL_ID, pdf_bytes)

    if not result_prebuilt.documents:
        logging.error("Neither model detected a document.")
        return {
            "filename": source_name.split("/")[-1],
            "fields": {},
            "_min_confidence": 0,
            "_avg_confidence": 0,
            "_model_used": "none",
            "exception_status": "Review",
            "exception_reason": "no_document_detected_by_either_model",
        }

    doc_prebuilt = result_prebuilt.documents[0]
    extracted_prebuilt = extract_fields(doc_prebuilt, source_name)
    extracted_prebuilt["_model_used"] = PREBUILT_MODEL_ID
    logging.info(
        f"Prebuilt model confidence: {extracted_prebuilt['_min_confidence']:.2f}"
    )
    return finalize(extracted_prebuilt)


def analyze_document(client, model_id, pdf_bytes):
    """Helper to call Document Intelligence and return result."""
    poller = client.begin_analyze_document(
        model_id=model_id,
        body=AnalyzeDocumentRequest(bytes_source=pdf_bytes)
    )
    return poller.result()


def finalize(extracted):
    """Apply validation and set exception status."""
    issues = validate(extracted)
    min_conf = extracted["_min_confidence"]

    if issues:
        extracted["exception_status"] = "Review"
        extracted["exception_reason"] = ", ".join(issues)
    elif min_conf < CONFIDENCE_THRESHOLD:
        extracted["exception_status"] = "Review"
        extracted["exception_reason"] = f"low_confidence ({min_conf:.2f})"
    else:
        extracted["exception_status"] = "OK"
        extracted["exception_reason"] = ""

    return extracted


def extract_fields(doc, filename):
    result = {
        "filename": filename.split("/")[-1],
        "doc_type_predicted": doc.doc_type,
        "fields": {},
    }
    confidences = []

    for field_name, field in (doc.fields or {}).items():
        if field is None:
            continue
        value = get_field_value(field)
        value = normalize_value(field_name, value)
        result["fields"][field_name] = {
            "value": value,
            "confidence": field.confidence,
        }
        if field.confidence is not None:
            confidences.append(field.confidence)

    result["_min_confidence"] = min(confidences) if confidences else 0
    result["_avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0
    return result


def get_field_value(field):
    """Pull the raw value out of a Document Intelligence field object."""
    if field.value_currency:
        return field.value_currency.amount
    if field.value_number is not None:
        return field.value_number
    if field.value_date:
        return str(field.value_date)
    if field.value_string:
        return field.value_string
    return field.content


def normalize_value(field_name, value):
    """
    Normalize amount fields to a clean POSITIVE float — strip any minus sign.
    '11.02-'  → 11.02
    '-97.44'  → 97.44
    -97.44    → 97.44   (DI may return a parsed negative float)
    '313.14'  → 313.14
    All other fields are returned untouched.
    """
    NUMERIC_FIELDS = {
        "InvoiceTotal",
        "SubTotal",
        "TotalTax",
        "AmountDue",
    }

    if field_name not in NUMERIC_FIELDS:
        return value  # leave everything else untouched

    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "")
        cleaned = cleaned.strip("-").strip()   # remove leading OR trailing minus
        try:
            return float(cleaned)              # always positive
        except ValueError:
            return value

    if isinstance(value, (int, float)):
        return abs(value)   # ← THE FIX: force positive (-97.44 → 97.44)

    return value

def validate(extracted):
    """Business-rule sanity checks independent of model confidence."""
    issues = []
    fields = extracted["fields"]

    def gv(name):
        return fields.get(name, {}).get("value")

    if not gv("InvoiceId"):
        issues.append("missing_invoice_number")
    if not gv("VendorName"):
        issues.append("missing_vendor_name")
    if not gv("InvoiceDate"):
        issues.append("missing_invoice_date")
    if gv("InvoiceTotal") is None:
        issues.append("missing_invoice_total")

    return issues


def flatten_for_power_automate(extracted):
    """
    Flatten the result into a simple flat object whose keys match
    what Power Automate maps to the Excel columns.
    """
    fields = extracted["fields"]

    def gv(name):
        return fields.get(name, {}).get("value")

    return {
        "InvoiceNumber": gv("InvoiceId"),
        "VendorName": gv("VendorName"),
        "InvoiceAmount": gv("InvoiceTotal"),
        "InvoiceDate": gv("InvoiceDate"),
        "ExceptionStatus": extracted.get("exception_status", "OK"),
        "ExceptionReason": extracted.get("exception_reason", ""),
        "MinConfidence": round(extracted.get("_min_confidence", 0), 3),
    }


# ── Blob output helpers (blob trigger only) ─────────────────────────
def write_result_to_blob(data, container_name, original_filename):
    if not STORAGE_CONN:
        return
    blob_service = BlobServiceClient.from_connection_string(STORAGE_CONN)
    container = blob_service.get_container_client(container_name)
    try:
        container.create_container()
    except Exception:
        pass
    base = original_filename.split("/")[-1].rsplit(".", 1)[0]
    container.upload_blob(
        name=f"{base}.json",
        data=json.dumps(data, indent=2, default=str),
        overwrite=True
    )


def write_error_to_blob(filename, error_msg):
    if not STORAGE_CONN:
        return
    blob_service = BlobServiceClient.from_connection_string(STORAGE_CONN)
    container = blob_service.get_container_client("invoices-errors")
    try:
        container.create_container()
    except Exception:
        pass
    base = filename.split("/")[-1].rsplit(".", 1)[0]
    container.upload_blob(
        name=f"{base}_error.json",
        data=json.dumps({"filename": filename, "error": error_msg}, indent=2),
        overwrite=True
    )


def _json_response(payload, status_code):
    return func.HttpResponse(
        json.dumps(payload, default=str),
        status_code=status_code,
        mimetype="application/json"
    )
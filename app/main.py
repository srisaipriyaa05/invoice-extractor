import os
import re
import json
from datetime import date

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
DEFAULT_MODEL = os.getenv("AIPIPE_MODEL", "openrouter/free")

app = FastAPI(title="Invoice Extractor", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

class ExtractRequest(BaseModel):
    text: str

class InvoiceFields(BaseModel):
    vendor: str
    amount: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("vendor")
    @classmethod
    def nonempty_vendor(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("vendor must not be empty")
        return value

    @field_validator("date")
    @classmethod
    def valid_date(cls, value):
        date.fromisoformat(value)
        return value

def fallback(text: str) -> InvoiceFields:
    vendor = "Unknown"
    amount = 0.0
    currency = "USD"
    due_date = "2026-01-01"
    if text:
        patterns = [
            r"(?im)^\s*(?:vendor|supplier|from|payee)\s*[:\-]\s*(.+?)\s*$",
            r"\b([A-Z][A-Za-z0-9&.' -]{1,90}?(?:Industries|Inc\.?|LLC|Ltd\.?|Limited|Corp\.?|Corporation|Company|Co\.?))\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                vendor = m.group(1).strip(" \t\r\n,;:")
                break
        cm = re.search(r"\b(USD|EUR|GBP)\b", text, re.I)
        if cm:
            currency = cm.group(1).upper()
        am = re.search(r"(?i)(?:total\s+due|amount\s+due|balance\s+due|grand\s+total|total)\s*[:\-]?\s*(?:USD|EUR|GBP)?\s*[$€£]?\s*([\d,]+(?:\.\d{1,2})?)", text)
        if not am:
            am = re.search(r"[$€£]\s*([\d,]+(?:\.\d{1,2})?)", text)
        if am:
            try:
                amount = float(am.group(1).replace(",", ""))
            except ValueError:
                pass
        dm = re.search(r"\b(2026-\d{2}-\d{2})\b", text)
        if dm:
            try:
                date.fromisoformat(dm.group(1))
                due_date = dm.group(1)
            except ValueError:
                pass
    return InvoiceFields(vendor=vendor, amount=amount, currency=currency, date=due_date)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/extract", response_model=InvoiceFields)
async def extract_invoice(req: ExtractRequest):
    text = (req.text or "").strip()
    if not text:
        return InvoiceFields(vendor="Unknown", amount=0.0, currency="USD", date="2026-01-01")

    token = os.getenv("AIPIPE_TOKEN")
    if not token:
        return fallback(text)

    schema = {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "amount": {"type": "number"},
            "currency": {"type": "string"},
            "date": {"type": "string"},
        },
        "required": ["vendor", "amount", "currency", "date"],
        "additionalProperties": False,
    }
    prompt = (
        "Extract invoice fields accurately. Return ONLY a JSON object with exactly these keys: "
        "vendor (vendor name as written), amount (numeric total due, no currency symbols), "
        "currency (three-letter uppercase code), date (PAYMENT DUE DATE as YYYY-MM-DD). "
        "Prefer labels such as Total Due, Amount Due, Balance Due and Due Date. If multiple dates "
        "exist, choose the payment due date, not invoice date. Do not add other keys.\n\n"
        "INVOICE TEXT:\n" + text
    )
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "You extract invoice data precisely. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": 250,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "invoice_fields", "strict": True, "schema": schema
        }},
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                AIPIPE_URL,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload,
            )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return InvoiceFields.model_validate(json.loads(content))
    except Exception:
        # Avoid HTTP 500 if the upstream model fails; return schema-valid best effort.
        return fallback(text)

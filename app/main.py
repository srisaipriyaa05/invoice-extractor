
import json
import os
import re
from datetime import date

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
DEFAULT_MODEL = os.getenv("AIPIPE_MODEL", "openrouter/free")

app = FastAPI(title="Invoice Extractor", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    text: str


class InvoiceFields(BaseModel):
    vendor: str
    amount: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("vendor")
    @classmethod
    def validate_vendor(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Vendor cannot be empty")
        return value

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value


def fallback(text: str) -> InvoiceFields:
    """Best-effort extraction if AI Pipe is unavailable."""
    vendor = "Unknown"
    amount = 0.0
    currency = "USD"
    due_date = "2026-01-01"

    if text:
        # Prefer an explicitly labelled vendor.
        match = re.search(
            r"(?im)^\s*(?:vendor|supplier|payee)\s*[:\-]\s*(.+?)\s*$",
            text,
        )

        if match:
            vendor = match.group(1).strip(" \t\r\n,;:")
        else:
            # Match the full company name, including its legal suffix.
            # Do not include phrases such as "Invoice from".
            org_pattern = (
                r"\b("
                r"(?:[A-Z][A-Za-z0-9&.'-]*\s+){0,7}"
                r"[A-Z][A-Za-z0-9&.'-]*"
                r"(?:\s+(?:Industries|Inc\.?|LLC|Ltd\.?|Limited|"
                r"Corp\.?|Corporation|Company|Co\.?))"
                r")(?=[\s,.;:]|$)"
            )

            match = re.search(org_pattern, text)
            if match:
                vendor = match.group(1).strip(" \t\r\n,;:")
            else:
                # Fallback for formats such as "Invoice from Acme Ltd."
                match = re.search(
                    r"(?i)\b(?:invoice\s+from|from)\s+"
                    r"(.+?)(?=\s+(?:invoice\s*(?:number|no\.?)|"
                    r"bill\s+to|total\s+due|amount\s+due|"
                    r"balance\s+due|due\s+date)\b|[,;\n]|$)",
                    text,
                )
                if match:
                    vendor = match.group(1).strip(" \t\r\n,;:")

    currency_match = re.search(r"\b(USD|EUR|GBP)\b", text, re.I)
    if currency_match:
        currency = currency_match.group(1).upper()

    amount_match = re.search(
        r"(?i)\b(?:total\s+due|amount\s+due|balance\s+due|"
        r"grand\s+total|total)\b\s*[:\-]?\s*"
        r"(?:(?:USD|EUR|GBP)\s*)?[$€£]?\s*"
        r"([\d,]+(?:\.\d{1,2})?)",
        text,
    )

    if not amount_match:
        amount_match = re.search(
            r"[$€£]\s*([\d,]+(?:\.\d{1,2})?)",
            text,
        )

    if amount_match:
        try:
            amount = float(amount_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Prefer an explicitly labelled payment due date.
    date_match = re.search(
        r"(?i)(?:payment\s+)?due\s+date\s*[:\-]?\s*"
        r"(2026-\d{2}-\d{2})",
        text,
    )

    if not date_match:
        date_match = re.search(r"\b(2026-\d{2}-\d{2})\b", text)

    if date_match:
        try:
            date.fromisoformat(date_match.group(1))
            due_date = date_match.group(1)
        except ValueError:
            pass

    return InvoiceFields(
        vendor=vendor,
        amount=amount,
        currency=currency,
        date=due_date,
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/extract", response_model=InvoiceFields)
async def extract_invoice(req: ExtractRequest):
    text = (req.text or "").strip()

    if not text:
        return InvoiceFields(
            vendor="Unknown",
            amount=0.0,
            currency="USD",
            date="2026-01-01",
        )

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
        "Extract the invoice fields accurately. Return only a JSON object "
        "with exactly these keys: vendor, amount, currency, date. "
        "The vendor must be the complete vendor name, without prefixes "
        "such as 'Invoice from'. Preserve identifiers and legal suffixes "
        "such as Ltd., LLC, Inc., and Industries. Amount must be the "
        "numeric total due without currency symbols. Currency must be "
        "a three-letter uppercase code. Date must be the PAYMENT DUE "
        "DATE in YYYY-MM-DD format. Prefer explicit Total Due, Amount "
        "Due, Balance Due, and Due Date labels. If multiple dates exist, "
        "choose the payment due date rather than the invoice date. "
        "Do not guess fields that are explicitly available in the text.\n\n"
        f"INVOICE TEXT:\n{text}"
    )

    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract invoice fields precisely. Return only "
                    "valid JSON matching the requested schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": 250,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "invoice_fields",
                "strict": True,
                "schema": schema,
            },
        },
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                AIPIPE_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]

        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )

        result = InvoiceFields.model_validate(json.loads(content))

        # Normalize currency and remove common vendor prefixes if the
        # model accidentally includes them.
        clean_vendor = re.sub(
            r"(?i)^\s*(?:invoice\s+from|vendor|supplier|payee)\s*[:\-]?\s*",
            "",
            result.vendor,
        ).strip()

        return InvoiceFields(
            vendor=clean_vendor or "Unknown",
            amount=result.amount,
            currency=result.currency.upper(),
            date=result.date,
        )

    except Exception:
        # Upstream errors must not result in HTTP 500.
        return fallback(text)

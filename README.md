# FastAPI Invoice Extractor

POST `/extract` with `{"text":"..."}`. Response contains exactly `vendor`, `amount`, `currency`, and `date`, validated with Pydantic.

## Deploy on Render
1. Upload these repository files to GitHub.
2. Create a Render Web Service from that repository using the Docker runtime.
3. Add `AIPIPE_TOKEN` in Render Environment settings (keep it secret).
4. Optionally set `AIPIPE_MODEL=openrouter/free`.
5. Use `https://YOUR-SERVICE.onrender.com/extract`.

The endpoint uses AI Pipe's OpenAI-compatible chat-completions API when configured. Empty input and upstream failures return a valid best-effort response rather than HTTP 500.

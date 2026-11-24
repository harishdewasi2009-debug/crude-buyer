# Autonomous Crude Oil Buyer Intelligence Agent

Researches Indian refinery procurement/tender pages and company sites, extracts **evidence**, classifies companies
A/B/C/D, finds public procurement contacts, and exports verified lists. It never sends offers or shares documents.

## Run
```
pip install -r requirements.txt
cp .env.example .env        # add API keys (any of the four providers) and optionally a search key
python run.py               # http://127.0.0.1:8000
python -m pytest tests      # 20 tests, no internet or API keys needed
```
Dashboard → *Find Buyers* → **Research all enabled sources + discovery**. Pause / Resume / Stop work live.

## How it decides (the important part)
| Class | Rule (enforced in code) |
|---|---|
| **A Confirmed** | verified purchase evidence **and** QC pass. Either (a) a producer's crude *sale/auction* award names the company as winner, (b) a refiner's own site publishes a crude *purchase* tender, or (c) an AI-extracted statement from the company's own site that a **different** AI model independently confirms. |
| **B Potential** | refinery operation, crude trading, crude-related award with unclear direction, or purchase sentence found by regex (not yet confirmed). |
| **C Vendor** | documented refinery award, non-crude, no purchase proof. |
| **D Not a buyer** | contractor / logistics / equipment (e.g. "crude oil *transportation*"). |

* **Direction matters.** If a *refiner* buys crude, the tender winner is its **supplier** (B at most, never "buyer").
  If a *producer* (ONGC, Oil India) sells crude, the winner is a **buyer**. Set each source's role under *Refinery Research*.
* **No hallucinated evidence.** Every AI-extracted claim must include a quote that is found **verbatim** in the stored text of a
  page the agent fetched; otherwise it is discarded. QC re-verifies all quotes independently.
* **No invented contacts.** Emails/phones/LinkedIn are copied from fetched pages; only emails on the company's own domain count
  as official; free-mail is flagged; names come from AI only with a verified quote. Everything else is `NOT FOUND`.
* **Class A is never hand-editable.** Only follow-up status and notes are.
* Broker/mandate language or a sanctions mention (high flag) blocks A → B.

## Modes
* **With ≥1 AI key:** AI extraction of awards/evidence/contacts, cross-model QC when ≥2 providers are set, fallback order, token/cost log.
* **With a search key (Brave or Google CSE):** private-buyer discovery from the open web.
* **No keys:** rule-based mode (table parsing + regex). Works, but A is limited to rule-checked official records.
  Costs show 0 until you enter your own prices under *AI Settings* (prices are not guessed).

## Conduct / safety
robots.txt honoured (unreadable robots = skip); ≥1 s per-host delay; login walls, CAPTCHAs, 401/403/429 are recorded and **not bypassed**;
private/internal addresses refused (SSRF guard); every page visit is logged. API keys live in environment variables only and are redacted
from errors. Dashboard binds to localhost, has CSRF header checks and optional `DASHBOARD_TOKEN`. Scraped text is HTML-escaped in the UI and
formula-neutralised in Excel/CSV.

## Known limits (be aware)
* Many refinery tender portals are JavaScript apps, behind login, or CAPTCHA-protected; those pages are skipped and logged,
  not forced. Add direct award-page URLs per source under *Extra tender-portal URLs*. Scanned PDFs are not OCR'd.
* MCA/CIN verification needs login/CAPTCHA, so the CIN is taken from the company's own pages and flagged for manual MCA confirmation.
* Source ownership/role defaults (HPCL, IOCL, … `Private`/`PSU`) are my starting values: check them.
* Heuristics can misread unusual tables. Treat output as **qualified leads to confirm**, and run your own sanctions/KYC screening before any dealing.
* Default AI model names are editable in *AI Settings*; check them against each provider's current list.

## Layout
`app/agents.py` (8 agents) · `app/director.py` (Agent 1, run control, scheduler) · `app/llm.py` (4 providers, fallback, cost) ·
`app/fetcher.py` · `app/db.py` (SQLite) · `app/exports.py` · `app/server.py` (API) · `app/static/index.html` (dashboard) · `tests/`.

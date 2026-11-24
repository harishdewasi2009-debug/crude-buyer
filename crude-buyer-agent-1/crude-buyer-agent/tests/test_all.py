import csv
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import exports                                    # noqa: E402
from app.agents import (Control, Ctx, Stopped, CompanyVerificationAgent, ContactFinderAgent, DueDiligenceAgent,  # noqa: E402
                        QualityControlAgent, DataManagerAgent, classify)
from app.db import DB                                      # noqa: E402
from app.director import Director                          # noqa: E402
from app.fetcher import Fetcher, Page, safe_url            # noqa: E402
from app.llm import LLM, LLMResult, parse_json             # noqa: E402
from app.server import create_app                          # noqa: E402
from app.util import name_key, quote_in_text, CIN_RE, email_is_official  # noqa: E402

TENDERS = """<html><title>Tender awards</title><body><h1>Tender Awards</h1><table>
<tr><th>Tender No</th><th>Description</th><th>Successful Bidder</th><th>Award Date</th></tr>
<tr><td>T-1</td><td>Sale of crude oil by auction (lifting)</td><td>Alpha Refining Pvt Ltd</td><td>2026-03-01</td></tr>
<tr><td>T-2</td><td>Transportation of crude oil by tanker</td><td>Blue Shipping Ltd</td><td>2026-02-10</td></tr>
<tr><td>T-3</td><td>Supply of valves for unit 4</td><td>Valve Works Ltd</td><td>2026-01-20</td></tr>
<tr><td>T-4</td><td>Crude oil import term contract purchase</td><td>Gamma Traders Pvt Ltd</td><td>2026-04-05</td></tr>
<tr><td>T-5</td><td>Supply of office stationery</td><td>Paper Co</td><td>2026-01-02</td></tr>
</table></body></html>"""
HOME = ("<html><title>Mock Upstream Ltd</title><body><h1>Mock Upstream</h1>"
        "<a href='/tenders'>Tender Awards</a> <a href='/private/tender-archive'>Tender archive</a> "
        "<a href='/about'>About us</a> <a href='/captcha'>Crude auction results</a> <a href='/loginpage'>Vendor portal</a>"
        "</body></html>")
ALPHA_HOME = ("<html><title>Alpha Refining</title><body>Welcome to Alpha Refining. "
              "<a href='/alpha/about'>About</a> <a href='/alpha/contact'>Contact</a></body></html>")
ALPHA_ABOUT = ("<html><body><p>Registered Office: 12 Marine Drive, Mumbai, Maharashtra 400001. "
               "CIN: L12345MH2000PLC123456</p>"
               "<p>The company imports crude oil from West Asia and Africa for processing at its refinery complex.</p>"
               "<p>Our refinery has a refining capacity of 20 MMTPA at the coastal site in Gujarat.</p></body></html>")
ALPHA_CONTACT = ("<html><body><p>Contact us. Write to procurement@alphaoil.in for tenders. Tel: +91 22 1234 5678.</p></body></html>")


class H(BaseHTTPRequestHandler):
    routes = {"/": HOME, "/tenders": TENDERS, "/about": "<html><body>About Mock Upstream. " + "x " * 10 + "</body></html>",
              "/alpha": ALPHA_HOME, "/alpha/about": ALPHA_ABOUT, "/alpha/contact": ALPHA_CONTACT,
              "/captcha": "<html><body>Please complete the CAPTCHA to continue</body></html>",
              "/loginpage": "<html><body><form><input type='password'></form>Login</body></html>",
              "/forbidden": "no"}
    hits = []

    def do_GET(self):
        H.hits.append(self.path)
        if self.path == "/robots.txt":
            body, code = "User-agent: *\nDisallow: /private/\n", 200
        elif self.path == "/forbidden":
            body, code = "nope", 403
        elif self.path in self.routes:
            body, code = self.routes[self.path], 200
        else:
            body, code = "not found", 404
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY",
              "BRAVE_API_KEY", "DASHBOARD_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CBA_ALLOW_LOCAL", "1")
    H.hits.clear()


@pytest.fixture
def db(tmp_path):
    d = DB(tmp_path / "t.db")
    d.set_setting("crawl_delay_sec", 0)
    return d


def mkctx(db, llm=None):
    ctl = Control()
    return Ctx(db, llm or LLM(db), Fetcher(db, ctl, 1), ctl, 1)


class FakeLLM:
    def __init__(self, db, providers=("openai", "anthropic"), cross=True, quote=None):
        self.db, self.providers, self.cross, self.quote, self.calls = db, list(providers), cross, quote, []

    def available(self):
        return self.providers

    def order(self, prefer=None, exclude=None):
        o = ([prefer] if prefer in self.providers else []) + [p for p in self.providers if p != prefer]
        return [p for p in o if p not in (exclude or [])]

    def complete(self, system, user, purpose, prefer=None, exclude=None, **kw):
        self.calls.append((purpose, prefer, exclude))
        data = {}
        if purpose == "verify-company":
            data = {"findings": [
                {"kind": "crude_purchase", "subject": "Alpha Refining", "quote": self.quote, "evidence_date": "2026"},
                {"kind": "crude_purchase", "subject": "Alpha Refining", "quote": "We import 500 million barrels from Mars every year.", "evidence_date": "2026"}]}
        elif purpose == "qc-cross-verify":
            data = {"supports": self.cross}
        return LLMResult(json.dumps(data), data, self.order(prefer, exclude)[0], "fake")


# ------------------------------------------------------------------ utils
def test_util_rules():
    assert name_key("Alpha Refining Pvt. Ltd.") == name_key("ALPHA REFINING PRIVATE LIMITED") == "alpha refining"
    assert CIN_RE.search("CIN L12345MH2000PLC123456 ok")
    assert quote_in_text("imports  crude oil from West Asia", "The company imports crude oil from West Asia and more")
    assert not quote_in_text("imports crude oil from Mars", "The company imports crude oil from West Asia")
    assert email_is_official("a@alphaoil.in", "https://www.alphaoil.in/")
    assert not email_is_official("a@gmail.com", "https://www.alphaoil.in/")
    assert parse_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_ssrf_guard(monkeypatch):
    monkeypatch.delenv("CBA_ALLOW_LOCAL")
    assert not safe_url("http://127.0.0.1:8000/")[0]
    assert not safe_url("file:///etc/passwd")[0]


# ------------------------------------------------------------------ fetcher politeness
def test_fetcher_respects_robots_captcha_login_403(db, server):
    f = Fetcher(db, None, 1)
    assert f.get(server + "/private/tender-archive").blocked.startswith("robots")
    assert "/private/tender-archive" not in H.hits
    assert "CAPTCHA" in f.get(server + "/captcha").blocked
    assert "login" in f.get(server + "/loginpage").blocked
    assert "403" in f.get(server + "/forbidden").blocked
    assert f.get(server + "/tenders").ok
    visits = {v["url"]: v for v in db.q("SELECT * FROM page_visits")}
    assert visits[server + "/private/tender-archive"]["allowed"] == 0


# ------------------------------------------------------------------ end to end without any AI key
def run_pipeline(db, server):
    db.ex("UPDATE sources SET enabled=0")
    db.ex("INSERT INTO sources(name,url,role,ownership) VALUES('Mock Upstream Ltd',?, 'producer','PSU')", (server + "/",))
    d = Director(db, LLM(db))
    d.start({})
    d.join(90)
    return d


def test_end_to_end_rule_mode(db, server):
    d = run_pipeline(db, server)
    run = db.one("SELECT * FROM runs ORDER BY id DESC LIMIT 1")
    assert run["status"] in ("completed", "completed_with_errors"), run
    cls = {c["name"]: c["classification"] for c in db.q("SELECT name,classification FROM companies")}
    assert cls["Alpha Refining Pvt Ltd"] == "A"          # producer SOLD crude to them -> documented buyer
    assert cls["Gamma Traders Pvt Ltd"] == "B"           # crude award, direction unclear -> potential only
    assert cls["Blue Shipping Ltd"] == "D"               # crude transport = logistics, not a buyer
    assert cls["Valve Works Ltd"] == "D"
    assert cls["Paper Co"] == "C"                        # refinery-linked vendor, no crude
    alpha = db.one("SELECT * FROM companies WHERE name LIKE 'Alpha%'")
    assert alpha["verification_status"] == "QC_PASSED"
    assert alpha["procurement_email"] is None and alpha["phone"] is None   # nothing invented
    assert "/private/tender-archive" not in H.hits       # robots honoured during the crawl
    assert db.one("SELECT COUNT(*) n FROM tenders")["n"] == 5
    assert db.one("SELECT COUNT(*) n FROM tasks WHERE status='failed'")["n"] == 0
    assert db.q("SELECT 1 FROM page_visits WHERE note LIKE '%CAPTCHA%'")
    ev = db.evidence_for(alpha["id"])
    assert ev and all(e["verified"] and e["url"].startswith(server) for e in ev)


# ------------------------------------------------------------------ agents with a fake AI
def _alpha(db, server):
    cid, _ = db.upsert_company("Alpha Refining Pvt Ltd", website=server + "/alpha", country="India")
    return cid


REAL = "The company imports crude oil from West Asia and Africa for processing at its refinery complex."


def _full(db, server, llm, cid):
    ctx = mkctx(db, llm)
    CompanyVerificationAgent(ctx).run(cid)
    ContactFinderAgent(ctx).run(cid)
    DueDiligenceAgent(ctx).run(cid)
    QualityControlAgent(ctx).run(cid)
    return db.company(cid)


def test_hallucinated_quote_is_rejected_and_cross_check_passes(db, server):
    cid = _alpha(db, server)
    c = _full(db, server, FakeLLM(db, cross=True, quote=REAL), cid)
    ev = db.evidence_for(cid)
    assert not any("Mars" in e["quote"] and e["verified"] for e in ev)
    assert any(e["kind"] == "crude_purchase" and e["verified"] and e["method"] == "llm" for e in ev)
    assert c["cin"] == "L12345MH2000PLC123456" and c["state"] == "Maharashtra"
    assert c["classification"] == "A" and c["verification_status"] == "QC_PASSED"


def test_cross_model_disagreement_downgrades(db, server):
    cid = _alpha(db, server)
    llm = FakeLLM(db, cross=False, quote=REAL)
    c = _full(db, server, llm, cid)
    assert c["classification"] == "B" and c["verification_status"] == "QC_DOWNGRADED"
    purposes = [x for x in llm.calls if x[0] == "qc-cross-verify"]
    assert purposes and purposes[0][1] == "anthropic" and purposes[0][2] == ["openai"]   # a *different* model judged it


def test_single_model_is_provisional_manual_review(db, server):
    cid = _alpha(db, server)
    c = _full(db, server, FakeLLM(db, providers=("openai",), quote=REAL), cid)
    assert c["classification"] == "A" and c["verification_status"] == "MANUAL_REVIEW"


def test_rule_mode_never_reaches_A_from_a_sentence(db, server):
    cid = _alpha(db, server)
    c = _full(db, server, LLM(db), cid)       # no keys: regex evidence only
    assert c["classification"] == "B"
    assert c["cin"] == "L12345MH2000PLC123456"


def test_contacts_only_official_domain(db):
    ctx = mkctx(db)
    cid, _ = db.upsert_company("Alpha Oil Ltd", website="https://www.alphaoil.in/")
    pg = Page("https://www.alphaoil.in/contact", ok=True, text="Email procurement@alphaoil.in or ceo@gmail.com or x@other.com Tel: +91 22 1234 5678",
              links=[("LinkedIn", "https://www.linkedin.com/company/alphaoil/"), ("Vendor registration", "https://www.alphaoil.in/vendor-registration")])
    found = {}
    ContactFinderAgent(ctx).harvest(pg, "https://www.alphaoil.in/", found)
    assert found["proc_email"] == ["procurement@alphaoil.in"] and "official_email" not in found
    assert found["linkedin"] and found["reg_link"] and found["phone"][0].startswith("+91")


def test_broker_language_blocks_class_A(db):
    cid, _ = db.upsert_company("Broker Oil Pvt Ltd", website="https://brokeroil.example/", country="India")
    url = "https://brokeroil.example/about"
    txt = "We act as a broker and mandate holder and also import crude oil for our own refinery operations."
    db.save_page(url, 200, "t", txt)
    db.add_evidence(cid, "crude_purchase", url, txt, "llm", "test")
    DueDiligenceAgent(mkctx(db)).run(cid)
    assert any(f["code"] == "INTERMEDIARY_LANGUAGE" for f in db.q("SELECT * FROM dd_flags WHERE company_id=?", (cid,)))
    QualityControlAgent(mkctx(db, FakeLLM(db))).run(cid)
    assert db.company(cid)["classification"] == "B"


def test_free_mail_flag(db):
    cid, _ = db.upsert_company("Gmail Oil", website="https://x.example/", procurement_email="deal@gmail.com")
    DueDiligenceAgent(mkctx(db)).run(cid)
    assert any(f["code"] == "FREE_MAIL" and f["severity"] == "high" for f in db.q("SELECT * FROM dd_flags"))


def test_unverified_evidence_does_not_classify(db):
    cid, _ = db.upsert_company("Ghost Oil")
    db.save_page("https://ghost.example/", 200, "t", "We sell candles and nothing else at all in this town.")
    db.add_evidence(cid, "crude_purchase", "https://ghost.example/", "We import crude oil in huge volumes every year", "llm", "t")
    assert classify(db.evidence_for(cid))[0] == "UNCLASSIFIED"


def test_dedupe_and_merge(db):
    a, _ = db.upsert_company("Alpha Oil Ltd", website="https://www.alphaoil.in/")
    b, created = db.upsert_company("Alpha Oil Limited")           # same name key -> same record
    assert a == b and not created
    c, created = db.upsert_company("Alpha Petroleum Trading", website="https://alphaoil.in/x")   # same domain -> same record
    assert c == a and not created
    x, _ = db.upsert_company("Beta One", cin="L12345MH2000PLC123456")
    y, created = db.upsert_company("Beta Uno", cin="L12345MH2000PLC123456")
    assert x == y and not created


def test_pause_resume_stop_control():
    ctl = Control()
    ctl.pause()
    out = []

    def work():
        try:
            ctl.checkpoint()
            out.append("ran")
        except Stopped:
            out.append("stopped")
    t = threading.Thread(target=work)
    t.start()
    time.sleep(0.3)
    assert out == []               # blocked while paused
    ctl.resume()
    t.join(2)
    assert out == ["ran"]
    ctl.pause()
    t = threading.Thread(target=work)
    t.start()
    time.sleep(0.2)
    ctl.stop()
    t.join(2)
    assert out[-1] == "stopped"


def test_director_pause_resume(db, server):
    db.ex("UPDATE sources SET enabled=0")
    db.ex("INSERT INTO sources(name,url,role,ownership) VALUES('Mock Upstream Ltd',?, 'producer','PSU')", (server + "/",))
    d = Director(db, LLM(db))
    d.start({})
    d.pause()
    assert db.one("SELECT status FROM runs ORDER BY id DESC LIMIT 1")["status"] == "paused"
    d.resume()
    d.join(90)
    assert db.one("SELECT status FROM runs ORDER BY id DESC LIMIT 1")["status"] in ("completed", "completed_with_errors")


def test_single_source_run_stays_scoped(db, server):
    db.ex("INSERT INTO sources(name,url,role,ownership) VALUES('Mock Upstream Ltd',?, 'producer','PSU')", (server + "/",))
    sid = db.one("SELECT id FROM sources WHERE name='Mock Upstream Ltd'")["id"]
    d = Director(db, LLM(db))
    d.start({"sources": [sid]})
    d.join(90)
    names = {c["name"] for c in db.q("SELECT name FROM companies")}
    assert "Mock Upstream Ltd" in names and "Alpha Refining Pvt Ltd" in names
    assert not any("HPCL" in n or "Reliance" in n for n in names)    # other seeded sources untouched


def test_restart_marks_stale_runs_interrupted(db):
    db.ex("INSERT INTO runs(status,config,started) VALUES('running','{}','x')")
    Director(db, LLM(db))
    assert db.one("SELECT status FROM runs")["status"] == "interrupted"


# ------------------------------------------------------------------ exports + API
def test_exports_and_api(tmp_path, server, monkeypatch):
    app = create_app(tmp_path / "data")
    db = app.config["DB"]
    db.set_setting("crawl_delay_sec", 0)
    db.upsert_company("=HYPERLINK(\"http://evil\")", country="India")      # formula injection attempt in scraped name
    run_pipeline(db, server)
    out = tmp_path / "out"
    made = exports.generate(db, out)
    assert set(made) == {"Indian Crude Oil Buyer Master List.xlsx", "Confirmed Buyers.csv", "Potential Buyers.csv",
                         "Refinery-Linked Vendors.csv", "Procurement Contacts.xlsx", "Company Due Diligence Reports.pdf"}
    rows = list(csv.reader(open(out / "Confirmed Buyers.csv", encoding="utf-8-sig")))
    assert rows[0][0] == "Company name" and [r[0] for r in rows[1:]].count("Alpha Refining Pvt Ltd") == 1
    assert all("Valve" not in r[0] and "Paper" not in r[0] for r in rows[1:])
    assert any("Gamma" in r[0] for r in csv.reader(open(out / "Potential Buyers.csv", encoding="utf-8-sig")))
    assert any("Paper Co" in r[0] for r in csv.reader(open(out / "Refinery-Linked Vendors.csv", encoding="utf-8-sig")))
    from openpyxl import load_workbook
    wb = load_workbook(out / "Indian Crude Oil Buyer Master List.xlsx")
    assert {"Primary Prospects (A+B)", "All Companies", "Tender Awards", "Evidence"} <= set(wb.sheetnames)
    allnames = [r[0].value for r in wb["All Companies"].iter_rows(min_row=2)]
    assert not any(str(n).startswith("=") for n in allnames)
    prim = {r[0].value for r in wb["Primary Prospects (A+B)"].iter_rows(min_row=2)}
    assert "Paper Co" not in prim and "Blue Shipping Ltd" not in prim           # only A and B
    pc = load_workbook(out / "Procurement Contacts.xlsx").active
    assert "NOT FOUND" in [c.value for r in pc.iter_rows(min_row=2) for c in r]
    assert (out / "Company Due Diligence Reports.pdf").read_bytes()[:4] == b"%PDF"

    c = app.test_client()
    H_ = {"X-Requested-With": "cba"}
    s = c.get("/api/summary").get_json()
    assert s["confirmed"] >= 1 and s["potential"] >= 1 and s["vendors"] == 1 and s["tasks_failed"] == 0
    assert all(x["classification"] == "A" for x in c.get("/api/companies?classification=A").get_json())
    assert c.get("/api/companies?primary=1").get_json()
    cid = c.get("/api/companies?q=Alpha").get_json()[0]["id"]
    assert c.patch(f"/api/companies/{cid}", json={"classification": "A"}, headers=H_).status_code == 400
    assert c.patch(f"/api/companies/{cid}", json={"follow_up_status": "Intro call booked"}, headers=H_).status_code == 200
    assert c.post("/api/run/start", json={}).status_code == 400                          # CSRF guard
    assert c.post("/api/settings", json={"crawl_delay_sec": 0}, headers=H_).status_code == 200
    assert c.get("/api/settings").get_json()["crawl_delay_sec"] == 1.0                   # clamped: never hammer sites
    d = c.get(f"/api/companies/{cid}").get_json()
    assert d["evidence"] and d["company"]["follow_up_status"] == "Intro call booked"
    assert c.get("/api/export/Confirmed Buyers.csv").status_code == 200
    assert c.get("/api/export/../../etc/passwd").status_code == 404
    assert c.get("/api/status").get_json()["active"] is False


def test_api_keys_never_exposed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-SUPERSECRET123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-SUPERSECRET456")
    app = create_app(tmp_path / "d2")
    c = app.test_client()
    blob = ""
    for p in ("/api/llm/status", "/api/settings", "/api/llm/usage", "/api/summary", "/api/status", "/"):
        blob += c.get(p).get_data(as_text=True)
    assert "SUPERSECRET" not in blob
    st = c.get("/api/llm/status").get_json()
    assert {p["provider"]: p["configured"] for p in st["providers"]} == {"openai": True, "anthropic": True, "gemini": False, "deepseek": False}
    assert LLM(app.config["DB"])._redact("boom sk-test-SUPERSECRET123") == "boom ***"


def test_llm_fallback_order_and_error_handling(db, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k2")
    llm = LLM(db)
    db.set_setting("primary_provider", "deepseek")
    assert llm.order() == ["deepseek", "openai"]
    calls = []

    def fake_call(p, system, user, mt):
        calls.append(p)
        if p == "deepseek":
            raise RuntimeError("HTTP 400 bad")
        return '{"ok": true}', 10, 5
    llm._call = fake_call
    db.set_setting("prices", {"openai": {"in": 1.0, "out": 2.0}})
    r = llm.complete("s", "u", "test")
    assert r.provider == "openai" and r.data == {"ok": True} and calls == ["deepseek", "openai"]
    u = db.q("SELECT provider, ok, cost FROM llm_usage ORDER BY id")
    assert [x["ok"] for x in u] == [0, 1] and abs(u[1]["cost"] - 0.00002) < 1e-9
    llm._call = lambda *a: (_ for _ in ()).throw(RuntimeError("HTTP 400 nope"))
    with pytest.raises(Exception):
        llm.complete("s", "u", "test")

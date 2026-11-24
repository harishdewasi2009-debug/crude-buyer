"""Flask backend + JSON API for the dashboard. API keys never leave the server."""
import json
import os
import secrets
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, send_file, abort, make_response

from . import exports
from .config import DEFAULT_DATA_DIR
from .db import DB, DEFAULT_SETTINGS
from .director import Director
from .llm import LLM, PROVIDERS


def create_app(data_dir=None, start_scheduler=False):
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    db = DB(data_dir / "buyers.db")
    llm = LLM(db)
    director = Director(db, llm)
    if start_scheduler:
        director.start_scheduler()
    app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
    if os.getenv("CBA_BEHIND_PROXY") == "1":   # Render terminates HTTPS and forwards the real scheme
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    app.config.update(DB=db, LLM=llm, DIRECTOR=director, EXPORT_DIR=data_dir / "exports")
    token = os.getenv("DASHBOARD_TOKEN")

    @app.before_request
    def guard():
        if request.path == "/login":
            return
        if token and not secrets.compare_digest(request.cookies.get("cba_token", ""), token):
            if request.path.startswith("/api/"):
                return jsonify(error="unauthorised"), 401
            return ("<form method=get action=/login>Access token: <input name=token type=password>"
                    "<button>Enter</button></form>"), 401
        if request.method in ("POST", "PATCH", "DELETE") and request.headers.get("X-Requested-With") != "cba":
            return jsonify(error="missing X-Requested-With header"), 400  # basic CSRF guard

    @app.after_request
    def headers(r):
        r.headers["X-Content-Type-Options"] = "nosniff"
        r.headers["X-Frame-Options"] = "DENY"
        r.headers["Cache-Control"] = "no-store"
        return r

    @app.get("/login")
    def login():
        resp = make_response("", 302)
        if token and secrets.compare_digest(request.args.get("token", ""), token):
            resp.set_cookie("cba_token", token, httponly=True, samesite="Strict", secure=request.is_secure)
        resp.headers["Location"] = "/"
        return resp

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    # ---------------------------------------------------------------- summary / status
    @app.get("/api/summary")
    def summary():
        n = lambda sql, a=(): db.one(sql, a)["n"]
        return jsonify(
            total=n("SELECT COUNT(*) n FROM companies"),
            confirmed=n("SELECT COUNT(*) n FROM companies WHERE classification='A'"),
            potential=n("SELECT COUNT(*) n FROM companies WHERE classification='B'"),
            vendors=n("SELECT COUNT(*) n FROM companies WHERE classification='C'"),
            not_buyers=n("SELECT COUNT(*) n FROM companies WHERE classification='D'"),
            verified_contacts=n("SELECT COUNT(*) n FROM companies WHERE contacts_verified=1 AND "
                                "(procurement_email IS NOT NULL OR procurement_contact IS NOT NULL)"),
            pending=n("SELECT COUNT(*) n FROM companies WHERE classification='UNCLASSIFIED' OR verification_status IN "
                      "('PENDING','EVIDENCE_COLLECTED','MANUAL_REVIEW')"),
            tasks_done=n("SELECT COUNT(*) n FROM tasks WHERE status IN ('completed','no_findings')"),
            tasks_failed=n("SELECT COUNT(*) n FROM tasks WHERE status='failed'"),
            tenders=n("SELECT COUNT(*) n FROM tenders"),
            llm_cost=db.one("SELECT COALESCE(SUM(cost),0) c FROM llm_usage")["c"],
        )

    @app.get("/api/status")
    def status():
        since = int(request.args.get("since", 0))
        run = db.one("SELECT * FROM runs ORDER BY id DESC LIMIT 1")
        return jsonify(run=run, active=director.active(), paused=bool(director.ctl and director.ctl.paused and director.active()),
                       tasks=db.q("SELECT * FROM tasks ORDER BY id DESC LIMIT 40"),
                       logs=db.q("SELECT * FROM logs WHERE id>? ORDER BY id DESC LIMIT 200", (since,)))

    # ---------------------------------------------------------------- run control
    @app.post("/api/run/start")
    def run_start():
        cfg = request.get_json(silent=True) or {}
        try:
            return jsonify(run_id=director.start({k: cfg.get(k) for k in ("sources", "discovery", "companies") if cfg.get(k)}))
        except RuntimeError as e:
            return jsonify(error=str(e)), 409

    for act in ("pause", "resume", "stop"):
        app.add_url_rule(f"/api/run/{act}", f"run_{act}", (lambda a=act: (getattr(director, a)(), jsonify(ok=True))[1]), methods=["POST"])

    # ---------------------------------------------------------------- data views
    @app.get("/api/companies")
    def companies():
        w, a = [], []
        for col, arg in (("classification", "classification"), ("state", "state"), ("verification_status", "verification"),
                         ("ownership", "ownership")):
            if request.args.get(arg):
                w.append(f"{col}=?")
                a.append(request.args[arg])
        if request.args.get("refinery"):
            w.append("refinery_connection LIKE ?")
            a.append(f"%{request.args['refinery']}%")
        if request.args.get("q"):
            w.append("(name LIKE ? OR notes LIKE ?)")
            a += [f"%{request.args['q']}%"] * 2
        if request.args.get("primary") == "1":
            w.append("classification IN ('A','B')")
        if request.args.get("category"):
            w.append("category LIKE ?")
            a.append(f"%{request.args['category']}%")
        sql = "SELECT * FROM companies" + (" WHERE " + " AND ".join(w) if w else "") + " ORDER BY classification, name LIMIT 1000"
        return jsonify(db.q(sql, tuple(a)))

    @app.get("/api/companies/<int:cid>")
    def company_detail(cid):
        c = db.company(cid) or abort(404)
        return jsonify(company=c, evidence=db.evidence_for(cid), flags=db.q("SELECT * FROM dd_flags WHERE company_id=?", (cid,)),
                       tenders=db.q("SELECT * FROM tenders WHERE company_id=?", (cid,)))

    @app.patch("/api/companies/<int:cid>")
    def company_patch(cid):
        b = request.get_json(silent=True) or {}
        upd = {k: str(b[k])[:2000] for k in ("follow_up_status", "notes") if k in b}
        if not upd:
            return jsonify(error="only follow_up_status and notes are editable; classification comes from evidence"), 400
        db.set_company(cid, **upd)
        return jsonify(ok=True)

    @app.post("/api/companies")
    def company_add():
        b = request.get_json(silent=True) or {}
        if not (b.get("name") or "").strip():
            return jsonify(error="name required"), 400
        cid, created = db.upsert_company(b["name"], website=b.get("website") or None, category="Manually added")
        return jsonify(id=cid, created=created)

    @app.get("/api/tenders")
    def tenders():
        return jsonify(db.q("SELECT * FROM tenders ORDER BY id DESC LIMIT 1000"))

    @app.get("/api/contacts")
    def contacts():
        return jsonify(db.q("SELECT id,name,classification,procurement_contact,procurement_email,official_email,phone,linkedin,"
                            "linkedin_people,supplier_reg_link,contact_source_url,contacts_verified,website FROM companies "
                            "WHERE classification IN ('A','B') ORDER BY classification,name"))

    @app.get("/api/flags")
    def flags():
        return jsonify(db.q("SELECT f.*, c.name, c.classification FROM dd_flags f JOIN companies c ON c.id=f.company_id "
                            "ORDER BY CASE f.severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, c.name LIMIT 1000"))

    @app.get("/api/history")
    def history():
        return jsonify(runs=db.q("SELECT * FROM runs ORDER BY id DESC LIMIT 50"),
                       pages=db.q("SELECT * FROM page_visits ORDER BY id DESC LIMIT 300"),
                       tasks=db.q("SELECT * FROM tasks ORDER BY id DESC LIMIT 300"))

    # ---------------------------------------------------------------- sources
    @app.get("/api/sources")
    def sources():
        return jsonify(db.q("SELECT * FROM sources ORDER BY id"))

    @app.post("/api/sources")
    def source_add():
        b = request.get_json(silent=True) or {}
        if not (b.get("name") and str(b.get("url", "")).startswith("http")):
            return jsonify(error="name and http(s) url required"), 400
        db.ex("INSERT OR IGNORE INTO sources(name,url,role,ownership,extra_urls) VALUES(?,?,?,?,?)",
              (b["name"], b["url"], b.get("role", "refiner") if b.get("role") in ("refiner", "producer") else "refiner",
               b.get("ownership", "Unknown"), json.dumps([u for u in b.get("extra_urls", []) if str(u).startswith("http")])))
        return jsonify(ok=True)

    @app.patch("/api/sources/<int:sid>")
    def source_patch(sid):
        b = request.get_json(silent=True) or {}
        for k in ("enabled", "role", "ownership", "url", "notes"):
            if k in b:
                db.ex(f"UPDATE sources SET {k}=? WHERE id=?", (b[k], sid))
        if "extra_urls" in b:
            db.ex("UPDATE sources SET extra_urls=? WHERE id=?", (json.dumps([u for u in b["extra_urls"] if str(u).startswith("http")]), sid))
        return jsonify(ok=True)

    # ---------------------------------------------------------------- AI settings (no keys exposed)
    @app.get("/api/llm/status")
    def llm_status():
        s = llm.status()
        s["prices"] = db.get_settings().get("prices")
        s["search_configured"] = bool(os.getenv("BRAVE_API_KEY") or (os.getenv("GOOGLE_CSE_KEY") and os.getenv("GOOGLE_CSE_CX")))
        return jsonify(s)

    @app.get("/api/llm/usage")
    def llm_usage():
        return jsonify(totals=db.q("SELECT provider, model, COUNT(*) calls, SUM(ok) ok_calls, SUM(in_tokens) in_tokens, "
                                   "SUM(out_tokens) out_tokens, ROUND(SUM(cost),4) cost FROM llm_usage GROUP BY provider, model"),
                       recent=db.q("SELECT * FROM llm_usage ORDER BY id DESC LIMIT 50"))

    @app.post("/api/llm/test")
    def llm_test():
        p = (request.get_json(silent=True) or {}).get("provider")
        if p not in PROVIDERS:
            return jsonify(error="unknown provider"), 400
        return jsonify(llm.test(p) if llm.key(p) else {"provider": p, "ok": False, "error": f"{PROVIDERS[p]['env'][0]} is not set on the server"})

    @app.get("/api/settings")
    def settings_get():
        return jsonify(db.get_settings())

    @app.post("/api/settings")
    def settings_set():
        b = request.get_json(silent=True) or {}
        changed = []
        for k, v in b.items():
            if k not in DEFAULT_SETTINGS:
                continue
            if k == "primary_provider" and v not in ("",) + tuple(PROVIDERS):
                return jsonify(error="bad provider"), 400
            if k == "fallback_order" and not (isinstance(v, list) and all(x in PROVIDERS for x in v)):
                return jsonify(error="bad fallback_order"), 400
            if k == "models" and not (isinstance(v, dict) and all(a in PROVIDERS and isinstance(m, str) for a, m in v.items())):
                return jsonify(error="bad models"), 400
            if k in ("crawl_delay_sec", "recheck_hours"):
                v = max(0.0, float(v))
                if k == "crawl_delay_sec":
                    v = max(v, 1.0)   # never hammer a site
            if k in ("max_pages_per_source", "max_depth", "max_llm_pages_per_source", "max_vendor_companies_per_source"):
                v = max(1, min(int(v), 500))
            db.set_setting(k, v)
            changed.append(k)
        return jsonify(changed=changed)

    # ---------------------------------------------------------------- exports
    @app.get("/api/export/<path:name>")
    def export(name):
        if name not in exports.EXPORTS:
            abort(404)
        exports.generate(db, app.config["EXPORT_DIR"], name)
        return send_file(app.config["EXPORT_DIR"] / name, as_attachment=True, download_name=name)

    @app.get("/api/exports")
    def export_list():
        return jsonify(list(exports.EXPORTS))

    return app

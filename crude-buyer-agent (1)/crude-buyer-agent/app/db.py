"""SQLite persistence shared by every agent (the 'common database')."""
import json
import sqlite3
import threading
import hashlib
from .util import now_iso, name_key, base_domain

COMPANY_COLS = ["country", "city", "state", "category", "ownership", "cin", "website", "refinery_connection",
                "purchase_evidence", "procurement_contact", "procurement_email", "official_email", "phone", "linkedin",
                "linkedin_people", "contact_source_url", "contacts_verified", "tender_reference", "award_date",
                "supplier_reg_link", "classification", "confidence", "buyer_type", "verification_status",
                "evidence_urls", "last_research_date", "follow_up_status", "notes", "registered_address"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies(
 id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, name_key TEXT NOT NULL,
 country TEXT, city TEXT, state TEXT, category TEXT, ownership TEXT, cin TEXT, website TEXT,
 refinery_connection TEXT, purchase_evidence TEXT, procurement_contact TEXT, procurement_email TEXT,
 official_email TEXT, phone TEXT, linkedin TEXT, linkedin_people TEXT, contact_source_url TEXT,
 contacts_verified INTEGER DEFAULT 0, tender_reference TEXT, award_date TEXT, supplier_reg_link TEXT,
 classification TEXT DEFAULT 'UNCLASSIFIED', confidence TEXT, buyer_type TEXT,
 verification_status TEXT DEFAULT 'PENDING', evidence_urls TEXT, last_research_date TEXT,
 follow_up_status TEXT DEFAULT 'Not contacted', notes TEXT, registered_address TEXT,
 created_at TEXT, updated_at TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS ux_company_key ON companies(name_key);
CREATE UNIQUE INDEX IF NOT EXISTS ux_company_cin ON companies(cin) WHERE cin IS NOT NULL;
CREATE TABLE IF NOT EXISTS evidence(
 id INTEGER PRIMARY KEY AUTOINCREMENT, company_id INTEGER NOT NULL, kind TEXT, url TEXT, quote TEXT,
 quote_hash TEXT, verified INTEGER DEFAULT 0, method TEXT, evidence_date TEXT, source_agent TEXT,
 run_id INTEGER, created_at TEXT, UNIQUE(company_id, kind, url, quote_hash));
CREATE TABLE IF NOT EXISTS tenders(
 id INTEGER PRIMARY KEY AUTOINCREMENT, source_name TEXT, tender_no TEXT, bidder TEXT, award_date TEXT,
 description TEXT, product_category TEXT, contract_value TEXT, duration TEXT, direction TEXT, kind TEXT,
 source_url TEXT, quote TEXT, quote_hash TEXT, verified INTEGER DEFAULT 0, company_id INTEGER, run_id INTEGER,
 created_at TEXT, UNIQUE(source_url, bidder, tender_no, quote_hash));
CREATE TABLE IF NOT EXISTS dd_flags(
 id INTEGER PRIMARY KEY AUTOINCREMENT, company_id INTEGER, code TEXT, severity TEXT, text TEXT, source_url TEXT,
 run_id INTEGER, created_at TEXT, UNIQUE(company_id, code, text));
CREATE TABLE IF NOT EXISTS pages(
 url TEXT PRIMARY KEY, status INTEGER, title TEXT, text TEXT, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS page_visits(
 id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, url TEXT, status INTEGER, allowed INTEGER, bytes INTEGER,
 note TEXT, purpose TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS runs(
 id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT, config TEXT, started TEXT, finished TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS tasks(
 id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, agent TEXT, type TEXT, target TEXT, status TEXT,
 started TEXT, finished TEXT, error TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS logs(
 id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, run_id INTEGER, level TEXT, agent TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS llm_usage(
 id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, provider TEXT, model TEXT, purpose TEXT, in_tokens INTEGER,
 out_tokens INTEGER, cost REAL, ok INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sources(
 id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, url TEXT, role TEXT, ownership TEXT,
 enabled INTEGER DEFAULT 1, extra_urls TEXT DEFAULT '[]', notes TEXT);
"""

# Starting sources: the websites named in the brief. Roles/ownership are editable in the dashboard.
DEFAULT_SOURCES = [
    ("Hindustan Petroleum Corporation Limited (HPCL)", "https://www.hindustanpetroleum.com/", "refiner", "PSU"),
    ("Indian Oil Corporation Limited (IOCL)", "https://iocl.com/", "refiner", "PSU"),
    ("Reliance Industries Limited", "https://www.ril.com/", "refiner", "Private"),
    ("Bharat Petroleum Corporation Limited (BPCL)", "https://www.bharatpetroleum.in/", "refiner", "PSU"),
    ("Nayara Energy Limited", "https://www.nayaraenergy.com/", "refiner", "Private"),
    ("Mangalore Refinery and Petrochemicals Limited (MRPL)", "https://www.mrpl.co.in/", "refiner", "PSU"),
    ("Chennai Petroleum Corporation Limited (CPCL)", "https://www.cpcl.co.in/", "refiner", "PSU"),
    ("HPCL-Mittal Energy Limited (HMEL)", "https://www.hmel.in/", "refiner", "JV (PSU + private)"),
    ("Oil and Natural Gas Corporation Limited (ONGC)", "https://www.ongcindia.com/", "producer", "PSU"),
    ("Oil India Limited", "https://www.oil-india.com/", "producer", "PSU"),
    ("Numaligarh Refinery Limited (NRL)", "https://www.nrl.co.in/", "refiner", "PSU"),
]

DEFAULT_SETTINGS = {
    "primary_provider": "",
    "fallback_order": ["openai", "anthropic", "gemini", "deepseek"],
    "models": {},
    "prices": {},  # {"openai": {"in": usd_per_1M, "out": usd_per_1M}} - set by you; unknown prices are not guessed
    "cross_verify": True,
    "crawl_delay_sec": 2.0,
    "max_pages_per_source": 40,
    "max_depth": 2,
    "max_llm_pages_per_source": 15,
    "max_vendor_companies_per_source": 200,
    "include_non_crude_vendors": True,
    "recheck_hours": 0,
    "search_queries": [
        "India private crude oil importer company",
        "India crude oil trading company import crude refinery supply",
        "India refinery crude oil procurement agreement private company",
        "India crude oil term contract refinery imports annual report",
    ],
}


def qhash(text):
    return hashlib.sha1((text or "").strip().lower().encode()).hexdigest()[:16]


class DB:
    def __init__(self, path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        self._seed_sources()

    # ---- generic helpers
    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        r = self.q(sql, args)
        return r[0] if r else None

    def ex(self, sql, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # ---- settings
    def get_settings(self):
        out = json.loads(json.dumps(DEFAULT_SETTINGS))
        for r in self.q("SELECT key,value FROM settings"):
            try:
                out[r["key"]] = json.loads(r["value"])
            except Exception:
                pass
        return out

    def set_setting(self, key, value):
        if key not in DEFAULT_SETTINGS:
            raise KeyError(key)
        self.ex("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)))

    # ---- sources
    def _seed_sources(self):
        if self.one("SELECT 1 AS x FROM sources LIMIT 1"):
            return
        for name, url, role, own in DEFAULT_SOURCES:
            self.ex("INSERT OR IGNORE INTO sources(name,url,role,ownership) VALUES(?,?,?,?)", (name, url, role, own))

    def ensure_source_companies(self, source_ids=None):
        """Every source is itself a candidate company (refiners are the direct buyers)."""
        ids = []
        if source_ids:
            rows = self.q(f"SELECT * FROM sources WHERE id IN ({','.join('?' * len(source_ids))})", tuple(source_ids))
        else:
            rows = self.q("SELECT * FROM sources WHERE enabled=1")
        for s in rows:
            cat = "Refinery operator" if s["role"] == "refiner" else "Crude producer / seller"
            cid, _ = self.upsert_company(s["name"], country="India", website=s["url"], category=cat,
                                         ownership=s["ownership"])
            ids.append(cid)
        return ids

    # ---- companies
    def upsert_company(self, name, **fields):
        name = (name or "").strip()
        key = name_key(name)
        if not key:
            raise ValueError("empty company name")
        bad = set(fields) - set(COMPANY_COLS)
        if bad:
            raise KeyError(bad)
        fields = {k: v for k, v in fields.items() if v not in (None, "")}
        with self.lock:
            row = self.one("SELECT * FROM companies WHERE name_key=?", (key,))
            if not row and fields.get("cin"):
                row = self.one("SELECT * FROM companies WHERE cin=?", (fields["cin"],))
            if not row and fields.get("website"):
                bd = base_domain(fields["website"])
                for r in self.q("SELECT id,website FROM companies WHERE website IS NOT NULL"):
                    if base_domain(r["website"]) == bd:
                        row = self.one("SELECT * FROM companies WHERE id=?", (r["id"],))
                        break
            if row:
                upd = {}
                for k, v in fields.items():
                    if k == "refinery_connection":
                        parts = [p.strip() for p in (row.get(k) or "").split(";") if p.strip()]
                        if v not in parts:
                            parts.append(v)
                        upd[k] = "; ".join(parts)
                    elif row.get(k) in (None, "") or k in ("classification", "confidence", "buyer_type",
                                                           "verification_status", "last_research_date",
                                                           "contacts_verified", "purchase_evidence",
                                                           "evidence_urls"):
                        upd[k] = v
                if upd:
                    upd["updated_at"] = now_iso()
                    self.ex(f"UPDATE companies SET {','.join(k + '=?' for k in upd)} WHERE id=?",
                            (*upd.values(), row["id"]))
                return row["id"], False
            cols = ["name", "name_key", "created_at", "updated_at"] + list(fields)
            vals = [name, key, now_iso(), now_iso()] + list(fields.values())
            cid = self.ex(f"INSERT INTO companies({','.join(cols)}) VALUES({','.join('?' * len(cols))})", vals)
            return cid, True

    def set_company(self, cid, **fields):
        bad = set(fields) - set(COMPANY_COLS) - {"name"}
        if bad:
            raise KeyError(bad)
        fields["updated_at"] = now_iso()
        self.ex(f"UPDATE companies SET {','.join(k + '=?' for k in fields)} WHERE id=?", (*fields.values(), cid))

    def company(self, cid):
        return self.one("SELECT * FROM companies WHERE id=?", (cid,))

    # ---- pages / evidence
    def save_page(self, url, status, title, text):
        self.ex("INSERT INTO pages(url,status,title,text,fetched_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(url) DO UPDATE SET status=excluded.status,title=excluded.title,text=excluded.text,"
                "fetched_at=excluded.fetched_at", (url, status, title, (text or "")[:300000], now_iso()))

    def page_text(self, url):
        r = self.one("SELECT text FROM pages WHERE url=?", (url,))
        return r["text"] if r else None

    def add_evidence(self, company_id, kind, url, quote, method, agent, run_id=None, evidence_date=None):
        """Evidence is 'verified' only if the quote is found in the stored text of the page we fetched."""
        from .util import quote_in_text
        text = self.page_text(url)
        verified = 1 if (text and quote_in_text(quote, text)) else 0
        try:
            self.ex("INSERT INTO evidence(company_id,kind,url,quote,quote_hash,verified,method,evidence_date,"
                    "source_agent,run_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (company_id, kind, url, (quote or "")[:600], qhash(quote), verified, method, evidence_date,
                     agent, run_id, now_iso()))
        except Exception:
            pass  # duplicate
        return bool(verified)

    def evidence_for(self, cid):
        return self.q("SELECT * FROM evidence WHERE company_id=? ORDER BY verified DESC, id", (cid,))

    def add_flag(self, cid, code, severity, text, url=None, run_id=None):
        try:
            self.ex("INSERT INTO dd_flags(company_id,code,severity,text,source_url,run_id,created_at) "
                    "VALUES(?,?,?,?,?,?,?)", (cid, code, severity, text, url, run_id, now_iso()))
        except Exception:
            pass

    # ---- logging
    def log(self, run_id, level, agent, message):
        self.ex("INSERT INTO logs(ts,run_id,level,agent,message) VALUES(?,?,?,?,?)",
                (now_iso(), run_id, level, agent, str(message)[:1500]))

    def visit(self, run_id, url, status, allowed, nbytes, note, purpose):
        self.ex("INSERT INTO page_visits(run_id,url,status,allowed,bytes,note,purpose,ts) VALUES(?,?,?,?,?,?,?,?)",
                (run_id, url, status, 1 if allowed else 0, nbytes, note, purpose, now_iso()))

"""The eight specialised agents. They only communicate through the shared database.

Evidence rules (enforced in code, not just in prompts):
  * a claim counts only if its quote is found verbatim in the stored text of a page we fetched;
  * contact details are copied from fetched pages, never generated or guessed;
  * class A needs verified purchase evidence AND a Quality Control pass.
"""
import difflib
import heapq
import json
import re
import threading
from urllib.parse import urlparse

from .llm import LLMError
from .util import (CIN_RE, EMAIL_RE, PHONE_RE, FREE_MAIL, INDIAN_STATES, clip, domain_of, email_is_official,
                   name_key, norm_text, quote_in_text, same_site, sentences, subject_matches, now_iso)


class Stopped(Exception):
    pass


class Control:
    def __init__(self):
        self._run = threading.Event()
        self._run.set()
        self.stop_flag = False

    def checkpoint(self):
        while not self._run.is_set():
            if self.stop_flag:
                raise Stopped()
            self._run.wait(0.4)
        if self.stop_flag:
            raise Stopped()

    def pause(self):
        self._run.clear()

    def resume(self):
        self._run.set()

    def stop(self):
        self.stop_flag = True
        self._run.set()

    @property
    def paused(self):
        return not self._run.is_set()


class Ctx:
    def __init__(self, db, llm, fetcher, ctl, run_id):
        self.db, self.llm, self.fetcher, self.ctl, self.run_id = db, llm, fetcher, ctl, run_id
        self.touched = set()

    def touch(self, cid):
        self.touched.add(cid)


class Agent:
    name = "Agent"

    def __init__(self, ctx):
        self.ctx, self.db = ctx, ctx.db

    def log(self, msg, level="info"):
        self.db.log(self.ctx.run_id, level, self.name, msg)

    def settings(self):
        return self.db.get_settings()

    def llm_ok(self):
        return bool(self.ctx.llm.available())

    def ask(self, system, user, purpose, **kw):
        try:
            return self.ctx.llm.complete(system, user, purpose, **kw)
        except LLMError as e:
            self.log(f"LLM unavailable for {purpose}: {e}", "warn")
            return None


# ---------------------------------------------------------------- classification helpers
CRUDE_RE = re.compile(r"\bcrude\b|refinery feedstock|petroleum procurement", re.I)
LOGISTIC_RE = re.compile(r"transport|freight|shipping|vessel|tanker|charter|haulage|logistic|storage|terminal|"
                         r"tank farm|jetty|\bspm\b|pipeline laying", re.I)
SERVICE_RE = re.compile(r"maintenance|repair|civil|construction|erection|fabrication|insulation|painting|scaffold|"
                        r"manpower|security|housekeeping|software|consultan|catalyst|valve|pump|equipment|"
                        r"engineering|\bepc\b|installation|canteen|\bamc\b|supply of (?:spares|material)", re.I)
BUY_DIR_RE = re.compile(r"\b(purchase|procurement|procure|import|buying|buy)\b", re.I)
SELL_DIR_RE = re.compile(r"\b(sale|selling|auction|disposal|lifting|e-?sale)\b", re.I)
BROKER_RE = re.compile(r"\b(broker|brokerage|mandate|facilitator|commission agent|on behalf of (?:the )?(?:buyer|seller)|"
                       r"intermediar(?:y|ies))\b", re.I)
SANCTION_RE = re.compile(r"\b(sanction(?:s|ed)?|ofac|sdn list)\b", re.I)

BUYER_KINDS = {"award_issuer_sells", "crude_purchase"}
POTENTIAL_KINDS = {"crude_purchase_candidate", "refinery_operation", "crude_trading", "award_issuer_buys",
                   "award_crude_unknown"}


def tender_direction(text, issuer_role):
    """Who is buying? 'issuer_sells' -> bidder is the buyer; 'issuer_buys' -> bidder is the supplier."""
    sell, buy = bool(SELL_DIR_RE.search(text)), bool(BUY_DIR_RE.search(text))
    if issuer_role == "producer" and sell and not buy:
        return "issuer_sells"
    if issuer_role == "refiner" and buy and not sell:
        return "issuer_buys"
    return "unknown"


def award_kind(text, direction):
    if CRUDE_RE.search(text):
        if LOGISTIC_RE.search(text):
            return "service"
        return {"issuer_sells": "award_issuer_sells", "issuer_buys": "award_issuer_buys"}.get(direction, "award_crude_unknown")
    return "service" if SERVICE_RE.search(text) or LOGISTIC_RE.search(text) else "refinery_vendor"


def classify(evidence):
    ver = [e for e in evidence if e["verified"]]
    kinds = {e["kind"] for e in ver}
    if kinds & BUYER_KINDS:
        cls = "A"
    elif kinds & POTENTIAL_KINDS:
        cls = "B"
    elif "service" in kinds:
        cls = "D"
    elif "refinery_vendor" in kinds:
        cls = "C"
    else:
        cls = "UNCLASSIFIED"
    urls = {e["url"] for e in ver if e["kind"] in BUYER_KINDS | POTENTIAL_KINDS}
    conf = "high" if len(urls) >= 2 else "medium" if urls else "low"
    return cls, conf, kinds


# ================================================================ AGENT 2
class RefineryProcurementAgent(Agent):
    name = "Agent 2 · Refinery Procurement"
    KW = [("award", 4), ("crude", 4), ("tender", 3), ("procure", 3), ("e-proc", 3), ("eproc", 3), ("auction", 3),
          ("vendor", 2), ("supplier", 2), ("contract", 2), ("purchase", 2), ("bid", 1), ("supply", 1), ("sale", 1),
          ("investor", 1), ("annual", 1)]
    SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".zip", ".mp4", ".mp3", ".doc", ".docx", ".xls", ".xlsx", ".svg")
    HDR = {"bidder": ["bidder", "awarded to", "awardee", "vendor", "supplier", "party", "contractor", "successful",
                      "l1", "name of the firm", "name of firm", "firm", "buyer", "lifter"],
           "tender": ["tender no", "tender number", "nit", "ref", "tender id", "enquiry", "rfq", "no."],
           "date": ["date"], "desc": ["description", "item", "subject", "work", "material", "tender name", "title",
                                      "product", "grade"],
           "value": ["value", "amount", "price"], "duration": ["period", "duration", "validity", "term"]}

    def score(self, text, href):
        s = (text + " " + href).lower()
        return sum(w for k, w in self.KW if k in s)

    def run(self, source):
        st = self.settings()
        base = source["url"]
        extras = json.loads(source.get("extra_urls") or "[]")
        heap, seen, pages, awards, llm_pages = [], set(), 0, 0, 0
        heapq.heappush(heap, (-10, 0, base))
        for u in extras:
            heapq.heappush(heap, (-9, 0, u))
        vendor_cap = int(st["max_vendor_companies_per_source"])
        vendors = 0
        issuer_id, _ = self.db.upsert_company(source["name"], country="India", website=base,
                                              category="Refinery operator" if source["role"] == "refiner" else "Crude producer / seller",
                                              ownership=source["ownership"])
        self.ctx.touch(issuer_id)
        while heap and pages < int(st["max_pages_per_source"]):
            self.ctx.ctl.checkpoint()
            _, depth, url = heapq.heappop(heap)
            if url in seen:
                continue
            seen.add(url)
            page = self.ctx.fetcher.get(url, purpose=f"{source['name']} procurement crawl")
            pages += 1
            if not page.ok:
                self.log(f"{url} not read: {page.blocked}", "warn")
                continue
            if depth < int(st["max_depth"]):
                for text, href in page.links:
                    if href in seen or href.lower().split("?")[0].endswith(self.SKIP_EXT):
                        continue
                    if not (same_site(href, base) or any(same_site(href, e) for e in extras)):
                        continue
                    sc = self.score(text, href)
                    if sc >= 2:
                        heapq.heappush(heap, (-sc, depth + 1, href))
                    if re.search(r"vendor|supplier", text + href, re.I) and re.search(r"regist|empanel|enrol|portal", text + href, re.I):
                        self.db.set_company(issuer_id, supplier_reg_link=href)
            rows = self.table_awards(page, source)
            if self.llm_ok() and llm_pages < int(st["max_llm_pages_per_source"]) and re.search(
                    r"award|successful bidder|work order|letter of acceptance|auction|crude", page.text, re.I) \
                    and CRUDE_RE.search(page.text):
                llm_pages += 1
                rows += self.llm_awards(page, source)
            for a in rows:
                if not a["crude"] and vendors >= vendor_cap:
                    continue
                if self.store_award(a, page, source, issuer_id):
                    awards += 1
                    vendors += 0 if a["crude"] else 1
        self.log(f"{source['name']}: {pages} pages read, {awards} award records stored")
        return {"pages": pages, "awards": awards, "findings": awards}

    def table_awards(self, page, source):
        out = []
        for rows in page.tables:
            hdr = [norm_text(h) for h in rows[0]]
            cols = {}
            for field, keys in self.HDR.items():
                for i, h in enumerate(hdr):
                    if any(k in h for k in keys):
                        cols.setdefault(field, i)
                        break
            if "bidder" not in cols:
                continue
            for r in rows[1:]:
                get = lambda f: (r[cols[f]] if f in cols and cols[f] < len(r) else "").strip()
                bidder = get("bidder")
                if not bidder or len(bidder) < 3:
                    continue
                joined = " ".join(r)
                desc = get("desc") or joined
                direction = tender_direction(joined, source["role"])
                out.append(dict(bidder=bidder, tender_no=get("tender"), award_date=get("date"), description=desc,
                                value=get("value"), duration=get("duration"), direction=direction,
                                quote=joined, method="table-rule", crude=bool(CRUDE_RE.search(joined)),
                                kind=award_kind(joined, direction)))
        return out

    def llm_awards(self, page, source):
        sys = ("You extract facts ONLY from the supplied page text. Never infer or invent. "
               "Return JSON: {\"awards\":[{\"bidder\":\"successful bidder legal name\",\"tender_no\":\"\","
               "\"award_date\":\"\",\"description\":\"\",\"value\":\"\",\"duration\":\"\","
               "\"direction\":\"issuer_buys|issuer_sells|unknown\",\"quote\":\"verbatim text <=300 chars\"}]}. "
               "direction=issuer_buys when the page issuer is purchasing/importing crude from the bidder; "
               "issuer_sells when the issuer is selling crude to the bidder. Only crude-oil/feedstock related awards.")
        r = self.ask(sys, f"Issuer: {source['name']} (role: {source['role']})\nURL: {page.url}\n\nTEXT:\n{clip(page.text, 14000)}",
                     "extract-awards")
        out = []
        for a in ((r.data or {}).get("awards") if r and isinstance(r.data, dict) else None) or []:
            if not isinstance(a, dict) or not a.get("bidder") or not a.get("quote"):
                continue
            if not quote_in_text(a["quote"], page.text):
                self.log(f"discarded award claim with unverifiable quote: {a.get('bidder')}", "warn")
                continue
            d = a.get("direction") if a.get("direction") in ("issuer_buys", "issuer_sells") else \
                tender_direction(a["quote"], source["role"])
            txt = f"{a.get('description', '')} {a['quote']}"
            out.append(dict(bidder=a["bidder"], tender_no=a.get("tender_no", ""), award_date=a.get("award_date", ""),
                            description=a.get("description", ""), value=a.get("value", ""), duration=a.get("duration", ""),
                            direction=d, quote=a["quote"], method="llm", crude=bool(CRUDE_RE.search(txt)),
                            kind=award_kind(txt, d)))
        return out

    def store_award(self, a, page, source, issuer_id):
        from .db import qhash
        cat = "Crude oil" if a["crude"] else "Non-crude / other"
        try:
            self.db.ex("INSERT INTO tenders(source_name,tender_no,bidder,award_date,description,product_category,"
                       "contract_value,duration,direction,kind,source_url,quote,quote_hash,verified,run_id,created_at) "
                       "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (source["name"], a["tender_no"], a["bidder"], a["award_date"], a["description"][:500], cat,
                        a["value"], a["duration"], a["direction"], a["kind"], page.url, a["quote"][:600],
                        qhash(a["quote"]), 1 if quote_in_text(a["quote"], page.text) else 0, self.ctx.run_id, now_iso()))
        except Exception:
            return False  # duplicate record
        cid, _ = self.db.upsert_company(a["bidder"], category="Tender bidder", refinery_connection=source["name"],
                                        tender_reference=a["tender_no"], award_date=a["award_date"])
        self.ctx.touch(cid)
        self.db.ex("UPDATE tenders SET company_id=? WHERE source_url=? AND bidder=? AND quote_hash=?",
                   (cid, page.url, a["bidder"], qhash(a["quote"])))
        self.db.add_evidence(cid, a["kind"], page.url, a["quote"], a["method"], self.name, self.ctx.run_id,
                             evidence_date=a["award_date"] or None)
        # A crude *purchase* tender on the issuer's own official site documents that the issuer buys crude.
        if a["kind"] == "award_issuer_buys":
            self.db.add_evidence(issuer_id, "crude_purchase", page.url, a["quote"], "table-rule" if a["method"] == "table-rule" else "llm",
                                 self.name, self.ctx.run_id, evidence_date=a["award_date"] or None)
        return True


# ================================================================ AGENT 3
class PrivateBuyerDiscoveryAgent(Agent):
    name = "Agent 3 · Private Buyer Discovery"

    def search(self, query):
        import os, requests
        if os.getenv("BRAVE_API_KEY"):
            r = requests.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": 10, "country": "IN"},
                             headers={"X-Subscription-Token": os.environ["BRAVE_API_KEY"], "Accept": "application/json"}, timeout=30)
            r.raise_for_status()
            return [x["url"] for x in r.json().get("web", {}).get("results", [])]
        if os.getenv("GOOGLE_CSE_KEY") and os.getenv("GOOGLE_CSE_CX"):
            r = requests.get("https://www.googleapis.com/customsearch/v1", params={
                "key": os.environ["GOOGLE_CSE_KEY"], "cx": os.environ["GOOGLE_CSE_CX"], "q": query, "num": 10}, timeout=30)
            r.raise_for_status()
            return [x["link"] for x in r.json().get("items", [])]
        return None

    def run_query(self, query):
        if not self.llm_ok():
            self.log("discovery skipped: needs an AI provider key to read pages", "warn")
            return {"findings": 0, "skipped": "no AI provider"}
        try:
            urls = self.search(query)
        except Exception as e:
            raise RuntimeError(f"search provider error: {type(e).__name__}")
        if urls is None:
            self.log("discovery skipped: set BRAVE_API_KEY or GOOGLE_CSE_KEY+GOOGLE_CSE_CX for web search", "warn")
            return {"findings": 0, "skipped": "no search provider"}
        found = 0
        sys = ("You extract facts ONLY from the supplied page text. Never infer. Return JSON: {\"companies\":["
               "{\"name\":\"\",\"website\":\"\",\"city\":\"\",\"kind\":\"crude_purchase|refinery_operation|crude_trading\","
               "\"quote\":\"verbatim <=300 chars proving the company ITSELF buys/imports/refines/trades crude oil\","
               "\"evidence_date\":\"YYYY or empty\"}]}. Only Indian companies. Refined-product (diesel, petrol) buyers "
               "do not qualify. Brokers/consultants that merely advise do not qualify.")
        for u in urls[:8]:
            self.ctx.ctl.checkpoint()
            pg = self.ctx.fetcher.get(u, purpose=f"discovery: {query}")
            if not pg.ok or not CRUDE_RE.search(pg.text):
                continue
            r = self.ask(sys, f"URL: {pg.url}\n\nTEXT:\n{clip(pg.text, 14000)}", "discover-companies")
            for c in ((r.data or {}).get("companies") if r and isinstance(r.data, dict) else None) or []:
                if not isinstance(c, dict) or not c.get("name") or not c.get("quote"):
                    continue
                if c.get("kind") not in ("crude_purchase", "refinery_operation", "crude_trading"):
                    continue
                if not quote_in_text(c["quote"], pg.text):
                    self.log(f"discarded unverifiable claim about {c.get('name')}", "warn")
                    continue
                cid, _ = self.db.upsert_company(c["name"], country="India", city=c.get("city") or None,
                                                category="Discovered (third-party source)")
                self.ctx.touch(cid)
                # third-party pages can support B at most until the company's own sources/QC confirm A
                kind = "crude_trading" if c["kind"] == "crude_purchase" else c["kind"]
                if self.db.add_evidence(cid, kind, pg.url, c["quote"], "llm", self.name, self.ctx.run_id,
                                        evidence_date=c.get("evidence_date") or None):
                    found += 1
                site = c.get("website") or ""
                if site.startswith("http") and not self.db.company(cid).get("website"):
                    self.db.set_company(cid, website=site, notes="website taken from third-party page; unverified")
        return {"findings": found}


# ================================================================ AGENT 4
class CompanyVerificationAgent(Agent):
    name = "Agent 4 · Company Verification"
    PAGE_KW = re.compile(r"about|contact|investor|annual|business|operation|trading|procure|supplier|vendor|refin|crude|"
                         r"company|corporate", re.I)
    PURCHASE = re.compile(r"\b(import(?:s|ed|ing)?|procure(?:s|d|ment)?|purchas(?:e|es|ed|ing)|sourc(?:e|es|ed|ing)|buy(?:s|ing)?)\b", re.I)
    OPERATES = re.compile(r"\b(our|its|the company'?s)\s+refiner(?:y|ies)\b|\b(operates?|owns?|commissioned)\b.{0,40}\brefiner|"
                          r"refining capacity|refinery capacity|\bMMTPA\b", re.I)
    TRADING = re.compile(r"crude.{0,40}\btrad(?:e|es|ing|er)\b|\btrad(?:e|es|ing|er)\b.{0,40}crude", re.I)

    def run(self, cid):
        c = self.db.company(cid)
        site = c.get("website")
        if not site:
            self.log(f"{c['name']}: no official website on record", "warn")
            self.db.set_company(cid, last_research_date=now_iso())
            return self.finish(cid, 0)
        home = self.ctx.fetcher.get(site, purpose=f"verify {c['name']}")
        pages = [home] if home.ok else []
        if not home.ok:
            self.log(f"{c['name']}: homepage not read ({home.blocked})", "warn")
        cand = []
        for text, href in (home.links if home.ok else []):
            if same_site(href, site) and self.PAGE_KW.search(text + " " + urlparse(href).path):
                cand.append(href)
        for href in list(dict.fromkeys(cand))[:6]:
            p = self.ctx.fetcher.get(href, purpose=f"verify {c['name']}")
            if p.ok:
                pages.append(p)
        n_ev = 0
        llm_left = 3
        for p in pages:
            self.ctx.ctl.checkpoint()
            self.facts_from_page(cid, c, p)
            for s in sentences(p.text):
                kind = None
                if CRUDE_RE.search(s) and self.PURCHASE.search(s):
                    kind = "crude_purchase_candidate"
                elif self.OPERATES.search(s):
                    kind = "refinery_operation"
                elif self.TRADING.search(s):
                    kind = "crude_trading"
                if kind and self.db.add_evidence(cid, kind, p.url, s, "regex", self.name, self.ctx.run_id):
                    n_ev += 1
            if llm_left and self.llm_ok() and CRUDE_RE.search(p.text):
                llm_left -= 1
                n_ev += self.llm_evidence(cid, c, p)
        self.db.set_company(cid, last_research_date=now_iso())
        return self.finish(cid, n_ev)

    def facts_from_page(self, cid, c, p):
        upd = {}
        m = CIN_RE.search(p.text)
        if m and not c.get("cin"):
            try:
                self.db.set_company(cid, cin=m.group(0), country="India")
            except Exception:
                self.log(f"CIN {m.group(0)} already belongs to another record; merge pending", "warn")
        low = p.text
        i = re.search(r"registered\s+(?:office|address)", low, re.I)
        if i and not c.get("registered_address"):
            chunk = low[i.end(): i.end() + 220].strip(" :-")
            upd["registered_address"] = chunk
            for st in INDIAN_STATES:
                if st.lower() in chunk.lower():
                    upd["state"] = st
                    break
        if upd:
            self.db.set_company(cid, **upd)

    def llm_evidence(self, cid, c, p):
        sys = ("You extract facts ONLY from the supplied page text. Never infer. Return JSON: {\"findings\":["
               "{\"kind\":\"crude_purchase|refinery_operation|crude_trading|service_provider\",\"subject\":\"company the "
               "sentence is about\",\"quote\":\"verbatim <=300 chars\",\"evidence_date\":\"YYYY or empty\"}]}. "
               "crude_purchase = the subject itself buys/imports/procures crude oil (own purchases). Purchases of "
               "diesel/petrol/other refined products are NOT crude purchases. Statements about other companies or the "
               "industry in general must name that other company as subject.")
        r = self.ask(sys, f"Company: {c['name']}\nURL: {p.url}\n\nTEXT:\n{clip(p.text, 14000)}", "verify-company")
        n = 0
        for f in ((r.data or {}).get("findings") if r and isinstance(r.data, dict) else None) or []:
            if not isinstance(f, dict) or not f.get("quote"):
                continue
            own = same_site(p.url, c.get("website") or "")
            subj = f.get("subject") or ""
            if not (subject_matches(subj, c["name"]) or (own and norm_text(subj) in ("the company", "we", "our company", "us", ""))):
                continue
            kind = {"service_provider": "service"}.get(f.get("kind"), f.get("kind"))
            if kind not in ("crude_purchase", "refinery_operation", "crude_trading", "service"):
                continue
            if kind == "crude_purchase" and not own:
                kind = "crude_trading"  # third-party statements cannot alone establish class A
            if self.db.add_evidence(cid, kind, p.url, f["quote"], "llm", self.name, self.ctx.run_id,
                                    evidence_date=f.get("evidence_date") or None):
                n += 1
        return n

    def finish(self, cid, n_ev):
        evs = self.db.evidence_for(cid)
        cls, conf, kinds = classify(evs)
        verified = [e for e in evs if e["verified"]]
        urls = sorted({e["url"] for e in verified})
        best = next((e for e in verified if e["kind"] in BUYER_KINDS), None) or \
            next((e for e in verified if e["kind"] in POTENTIAL_KINDS), None) or (verified[0] if verified else None)
        btype = "Unknown"
        if "crude_purchase" in kinds or "award_issuer_sells" in kinds:
            btype = "Direct buyer"
        elif "crude_purchase_candidate" in kinds or "refinery_operation" in kinds:
            btype = "Refinery operator (purchase evidence to confirm)"
        elif kinds & {"crude_trading", "award_issuer_buys", "award_crude_unknown"}:
            btype = "Trader / supplier-side (buying role unconfirmed)"
        self.db.set_company(cid, classification=cls, confidence=conf, buyer_type=btype,
                            verification_status="EVIDENCE_COLLECTED" if verified else "UNVERIFIED",
                            evidence_urls=json.dumps(urls),
                            purchase_evidence=(best["quote"][:400] if best else None))
        return {"findings": len(verified), "classification": cls}


# ================================================================ AGENT 5
class ContactFinderAgent(Agent):
    name = "Agent 5 · Contact Finder"
    PROC_LOCAL = re.compile(r"procure|tender|purchas|vendor|supply|sourcing|crude|trading|commercial|contract", re.I)

    def run(self, cid):
        c = self.db.company(cid)
        site = c.get("website")
        if not site:
            return {"findings": 0}
        urls = [site]
        home = self.ctx.fetcher.get(site, purpose=f"contacts {c['name']}")
        if home.ok:
            for t, h in home.links:
                if same_site(h, site) and re.search(r"contact|procure|vendor|supplier|tender|about|reach|enquir", t + urlparse(h).path, re.I):
                    urls.append(h)
        found = {}
        pages = []
        for u in list(dict.fromkeys(urls))[:6]:
            pg = self.ctx.fetcher.get(u, purpose=f"contacts {c['name']}")
            if not pg.ok:
                continue
            pages.append(pg)
            self.harvest(pg, site, found)
        if self.llm_ok():
            for pg in pages[:3]:
                if re.search(r"procurement|purchase|sourcing|commercial|trading|vendor", pg.text, re.I):
                    self.people(pg, c, found)
        upd = {}
        if found.get("proc_email"):
            upd["procurement_email"] = found["proc_email"][0]
        if found.get("official_email"):
            upd["official_email"] = found["official_email"][0]
        if found.get("phone"):
            upd["phone"] = found["phone"][0]
        if found.get("linkedin"):
            upd["linkedin"] = found["linkedin"][0]
        if found.get("people"):
            upd["procurement_contact"] = found["people"][0]
        if found.get("li_people"):
            upd["linkedin_people"] = json.dumps(found["li_people"][:5])
        if found.get("reg_link") and not c.get("supplier_reg_link"):
            upd["supplier_reg_link"] = found["reg_link"][0]
        if upd:
            upd["contact_source_url"] = found.get("source")
            upd["contacts_verified"] = 1 if (upd.get("procurement_email") or upd.get("official_email") or upd.get("phone")) else 0
            self.db.set_company(cid, **upd)
        return {"findings": len([k for k in ("procurement_email", "official_email", "phone", "linkedin", "procurement_contact") if k in upd])}

    def harvest(self, pg, site, found):
        emails = set(EMAIL_RE.findall(pg.text))
        for t, h in pg.links:
            if h.lower().startswith("mailto:"):
                emails.update(EMAIL_RE.findall(h))
        for e in sorted(emails):
            if not email_is_official(e, site):
                continue  # third-party / free-mail addresses are never recorded as official
            found.setdefault("source", pg.url)
            key = "proc_email" if self.PROC_LOCAL.search(e.split("@")[0]) else "official_email"
            found.setdefault(key, []).append(e.lower())
        for t, h in pg.links:
            hl = h.lower()
            if hl.startswith("tel:"):
                found.setdefault("phone", []).append(re.sub(r"[^\d+]", "", h[4:]) and h[4:].strip())
                found.setdefault("source", pg.url)
            if "linkedin.com/company/" in hl:
                found.setdefault("linkedin", []).append(h.split("?")[0])
            elif "linkedin.com/in/" in hl:
                found.setdefault("li_people", []).append({"url": h.split("?")[0], "label": t})
            if re.search(r"vendor|supplier", t + h, re.I) and re.search(r"regist|empanel|enrol|portal", t + h, re.I):
                found.setdefault("reg_link", []).append(h)
        for m in PHONE_RE.finditer(pg.text):
            found.setdefault("phone", []).append(m.group(1).strip())
            found.setdefault("source", pg.url)

    def people(self, pg, c, found):
        sys = ("Extract named people who hold procurement / crude sourcing / commercial / trading roles, ONLY from the "
               "text. Return JSON {\"people\":[{\"name\":\"\",\"title\":\"\",\"quote\":\"verbatim <=250 chars containing the name\"}]}. "
               "Never guess. Empty list if none.")
        r = self.ask(sys, f"Company: {c['name']}\nURL: {pg.url}\n\nTEXT:\n{clip(pg.text, 12000)}", "find-contacts")
        for p in ((r.data or {}).get("people") if r and isinstance(r.data, dict) else None) or []:
            if isinstance(p, dict) and p.get("name") and p.get("quote") and quote_in_text(p["quote"], pg.text) \
                    and norm_text(p["name"]) in norm_text(pg.text):
                found.setdefault("people", []).append(f"{p['name']} — {p.get('title', '')}".strip(" —"))
                found.setdefault("source", pg.url)


# ================================================================ AGENT 6
class DueDiligenceAgent(Agent):
    name = "Agent 6 · Due Diligence"

    def run(self, cid):
        c = self.db.company(cid)
        evs = self.db.evidence_for(cid)
        rid = self.ctx.run_id
        self.db.ex("DELETE FROM dd_flags WHERE company_id=?", (cid,))   # re-evaluated from current evidence
        add = lambda code, sev, text, url=None: self.db.add_flag(cid, code, sev, text, url, rid)
        if not c.get("cin"):
            add("NO_CIN", "medium", "CIN not found on official pages; Indian registration unverified (check MCA manually - MCA requires login/CAPTCHA so it is not automated).")
        if (c.get("country") or "").lower() not in ("india",):
            add("COUNTRY", "medium", "Indian incorporation not established (no CIN found).")
        if not c.get("website"):
            add("NO_WEBSITE", "medium", "No official website recorded.")
        unver = [e for e in evs if not e["verified"]]
        if unver:
            add("UNVERIFIED_CLAIMS", "low", f"{len(unver)} claim(s) discarded: quote not found in the fetched page text.")
        for e in evs:
            if e["verified"] and BROKER_RE.search(e["quote"] or ""):
                add("INTERMEDIARY_LANGUAGE", "high", "Evidence text uses broker/mandate/intermediary language; confirm this company is the end buyer.", e["url"])
            if e["verified"] and SANCTION_RE.search(e["quote"] or ""):
                add("SANCTIONS_MENTION", "high", "Evidence mentions sanctions; run formal sanctions/KYC screening before any dealing.", e["url"])
            m = re.search(r"(20\d{2})", e.get("evidence_date") or "")
            if m and int(m.group(1)) < int(now_iso()[:4]) - 3 and e["kind"] in BUYER_KINDS | POTENTIAL_KINDS:
                add("STALE_EVIDENCE", "low", f"Evidence dated {m.group(1)} is more than 3 years old.", e["url"])
        site_text = " ".join(self.db.page_text(u) or "" for u in json.loads(c.get("evidence_urls") or "[]")[:5])
        if c.get("website"):
            hp = self.db.page_text(c["website"]) or ""
            if BROKER_RE.search(hp):
                add("INTERMEDIARY_LANGUAGE", "high", "Company website uses broker/mandate/intermediary language; confirm it is an end buyer.", c["website"])
            site_text += " " + hp
        if SANCTION_RE.search(site_text):
            add("SANCTIONS_MENTION", "medium", "Pages reviewed mention sanctions; perform formal sanctions screening.")
        cat = (c.get("category") or "")
        if "Refinery" in cat and not any(e["verified"] and e["kind"] in POTENTIAL_KINDS | BUYER_KINDS for e in evs):
            add("CATEGORY_MISMATCH", "medium", "Recorded as a refinery operator but no verified refinery/crude evidence found.")
        for em in (c.get("procurement_email"), c.get("official_email")):
            if em and em.split("@")[-1] in FREE_MAIL:
                add("FREE_MAIL", "high", f"Contact uses free-mail domain ({em.split('@')[-1]}).")
        if c.get("supplier_reg_link") and not any(e["verified"] and e["kind"] in BUYER_KINDS for e in evs):
            add("VENDOR_REG_ONLY", "low", "A supplier-registration link exists; vendor registration does not prove crude purchasing authority.", c["supplier_reg_link"])
        for o in self.db.q("SELECT id,name FROM companies WHERE id<>?", (cid,)):
            if difflib.SequenceMatcher(None, name_key(o["name"]), name_key(c["name"])).ratio() > 0.9:
                add("POSSIBLE_DUPLICATE", "low", f"Name very similar to record #{o['id']} '{o['name']}'.")
        n = self.db.one("SELECT COUNT(*) AS n FROM dd_flags WHERE company_id=?", (cid,))["n"]
        return {"findings": n}


# ================================================================ AGENT 8
class QualityControlAgent(Agent):
    name = "Agent 8 · Quality Control"

    def run(self, cid):
        c = self.db.company(cid)
        evs = self.db.evidence_for(cid)
        cls, conf, kinds = classify(evs)
        notes = []
        status = "QC_PASSED" if cls != "UNCLASSIFIED" else "UNVERIFIED"
        # re-verify every quote against stored page text, independent of the agent that collected it
        for e in evs:
            ok = bool(quote_in_text(e["quote"], self.db.page_text(e["url"]) or ""))
            if ok != bool(e["verified"]):
                self.db.ex("UPDATE evidence SET verified=? WHERE id=?", (int(ok), e["id"]))
        evs = self.db.evidence_for(cid)
        cls, conf, kinds = classify(evs)
        if cls == "A":
            flags = self.db.q("SELECT * FROM dd_flags WHERE company_id=? AND severity='high'", (cid,))
            buyer_ev = [e for e in evs if e["verified"] and e["kind"] in BUYER_KINDS]
            if flags:
                cls, status = "B", "QC_DOWNGRADED"
                notes.append("Downgraded A→B: unresolved high-severity due-diligence flag(s): " + "; ".join(f["code"] for f in flags))
            else:
                llm_ev = [e for e in buyer_ev if e["method"] == "llm"]
                rule_ev = [e for e in buyer_ev if e["method"] != "llm"]
                if rule_ev:
                    notes.append("A supported by an official-site record (rule-checked).")
                elif llm_ev:
                    verdict = self.cross_check(c, llm_ev[0])
                    if verdict is True:
                        notes.append("Independent AI cross-check agreed.")
                    elif verdict is False:
                        cls, status = "B", "QC_DOWNGRADED"
                        notes.append("Downgraded A→B: independent AI cross-check did not support direct purchasing.")
                    else:
                        status = "MANUAL_REVIEW"
                        conf = "medium" if conf == "high" else conf
                        notes.append("No independent model available: A is provisional, manual review recommended.")
        if cls in ("UNCLASSIFIED",):
            notes.append("No verified evidence.")
        old = (c.get("notes") or "").split("\n[QC]")[0]
        self.db.set_company(cid, classification=cls, confidence=conf, verification_status=status,
                            notes=(old + "\n[QC] " + " ".join(notes)).strip() if notes else old or None)
        return {"findings": 1 if cls != "UNCLASSIFIED" else 0, "classification": cls, "status": status}

    def cross_check(self, c, ev):
        st = self.settings()
        if not st.get("cross_verify") or not self.llm_ok():
            return None
        text = self.db.page_text(ev["url"]) or ""
        i = norm_text(text).find(norm_text(ev["quote"])[:60])
        excerpt = text[max(0, i - 1200): i + 1800] if i >= 0 else text[:3000]
        first = self.ctx.llm.order()[0] if self.ctx.llm.order() else None
        others = [p for p in self.ctx.llm.order() if p != first]
        if not others:
            return None
        sys = ("You are an independent auditor. Judge ONLY from the supplied text. Return JSON "
               "{\"supports\":true|false,\"reason\":\"\"}. supports=true only if the text shows the named company "
               "ITSELF buys/imports/procures CRUDE OIL (not refined products, not services to a refinery).")
        r = self.ask(sys, f"Company: {c['name']}\nClaimed evidence: {ev['quote']}\n\nPAGE EXCERPT:\n{excerpt}",
                     "qc-cross-verify", prefer=others[0], exclude=[first])
        if r and isinstance(r.data, dict) and isinstance(r.data.get("supports"), bool):
            return r.data["supports"]
        return None


# ================================================================ AGENT 7
class DataManagerAgent(Agent):
    name = "Agent 7 · Data Manager"

    def dedupe(self):
        merged = 0
        rows = self.db.q("SELECT * FROM companies ORDER BY id")
        from .util import base_domain
        keep = {}
        for r in rows:
            sig = [("cin", r["cin"])] if r["cin"] else []
            if r["website"]:
                sig.append(("dom", base_domain(r["website"])))
            hit = next((keep[s] for s in sig if s in keep), None)
            if hit and hit != r["id"]:
                self.merge(hit, r["id"])
                merged += 1
                continue
            for s in sig:
                keep.setdefault(s, r["id"])
        if merged:
            self.log(f"merged {merged} duplicate company record(s)")
        return merged

    def merge(self, keep_id, drop_id):
        k, d = self.db.company(keep_id), self.db.company(drop_id)
        fill = {col: d[col] for col in d if col in ("country", "city", "state", "cin", "website", "category", "ownership",
                                                    "procurement_email", "official_email", "phone", "linkedin",
                                                    "supplier_reg_link", "tender_reference", "award_date") and d[col] and not k.get(col)}
        conn = d.get("refinery_connection") or ""
        for part in [p.strip() for p in conn.split(";") if p.strip()]:
            if part not in (k.get("refinery_connection") or ""):
                fill["refinery_connection"] = ((k.get("refinery_connection") or "") + "; " + part).strip("; ")
        self.db.ex("UPDATE OR IGNORE evidence SET company_id=? WHERE company_id=?", (keep_id, drop_id))
        self.db.ex("DELETE FROM evidence WHERE company_id=?", (drop_id,))
        self.db.ex("UPDATE tenders SET company_id=? WHERE company_id=?", (keep_id, drop_id))
        self.db.ex("DELETE FROM dd_flags WHERE company_id=?", (drop_id,))
        self.db.ex("DELETE FROM companies WHERE id=?", (drop_id,))
        if fill:
            self.db.set_company(keep_id, **fill)

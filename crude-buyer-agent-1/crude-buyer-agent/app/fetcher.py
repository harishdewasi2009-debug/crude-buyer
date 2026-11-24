"""Polite web fetcher: robots.txt, per-host rate limit, SSRF guard, no login/CAPTCHA bypass."""
import io
import ipaddress
import os
import re
import socket
import time
import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup

from .config import USER_AGENT, UA_TOKEN

BLOCK_RE = re.compile(r"captcha|are you (a )?(robot|human)|verify you are human|unusual traffic|access denied|"
                      r"just a moment\.\.\.|attention required", re.I)


@dataclass
class Page:
    url: str
    ok: bool = False
    status: int = 0
    title: str = ""
    text: str = ""
    links: list = field(default_factory=list)   # [(anchor text, absolute href)]
    tables: list = field(default_factory=list)  # [[ [cell,...], ...rows ]]
    blocked: str = ""                           # reason we did not/could not read it
    content_type: str = ""


def safe_url(url):
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False, "unsupported URL scheme"
    if os.getenv("CBA_ALLOW_LOCAL") == "1":
        return True, ""
    try:
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False, "refused: private/internal address"
    except socket.gaierror:
        return False, "DNS lookup failed"
    return True, ""


class Fetcher:
    def __init__(self, db, ctl=None, run_id=None):
        self.db, self.ctl, self.run_id = db, ctl, run_id
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.8"})
        self._robots, self._last, self._cache = {}, {}, {}

    def _delay(self):
        return float(self.db.get_settings().get("crawl_delay_sec", 2.0))

    def _robots_ok(self, url):
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.s.get(origin + "/robots.txt", timeout=15)
                if r.status_code >= 500:
                    rp = None          # cannot read policy -> be conservative
                elif r.status_code >= 400:
                    rp.parse([])       # no robots file -> allowed
                else:
                    rp.parse(r.text.splitlines())
            except requests.RequestException:
                rp = None
            self._robots[origin] = rp
        rp = self._robots[origin]
        return bool(rp and rp.can_fetch(UA_TOKEN, url))

    def _throttle(self, host):
        wait = self._delay() - (time.time() - self._last.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.time()

    def get(self, url, purpose=""):
        if self.ctl:
            self.ctl.checkpoint()
        url = urldefrag(url)[0]
        if url in self._cache:
            return self._cache[url]
        page = self._fetch(url, purpose)
        self._cache[url] = page
        return page

    def _fetch(self, url, purpose):
        cur = url
        r = None
        for _ in range(6):
            ok, why = safe_url(cur)
            if not ok:
                return self._done(Page(url, blocked=why), purpose, 0, False, 0)
            if not self._robots_ok(cur):
                return self._done(Page(url, blocked="robots.txt disallows or is unreadable"), purpose, 0, False, 0)
            self._throttle(urlparse(cur).netloc)
            try:
                r = self.s.get(cur, timeout=25, stream=True, allow_redirects=False)
            except requests.RequestException as e:
                return self._done(Page(url, blocked=f"network error: {type(e).__name__}"), purpose, 0, True, 0)
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                cur = urljoin(cur, r.headers["location"])
                continue
            break
        else:
            return self._done(Page(url, blocked="too many redirects"), purpose, 0, True, 0)

        raw = r.raw.read(3_000_000, decode_content=True)
        ctype = r.headers.get("content-type", "").lower()
        pg = Page(cur, status=r.status_code, content_type=ctype)
        if r.status_code in (401, 407):
            pg.blocked = "login required (not bypassed)"
        elif r.status_code in (403, 429):
            pg.blocked = f"access refused (HTTP {r.status_code}); not bypassed"
        elif r.status_code >= 400:
            pg.blocked = f"HTTP {r.status_code}"
        elif "pdf" in ctype or cur.lower().split("?")[0].endswith(".pdf"):
            self._parse_pdf(pg, raw)
        elif "html" in ctype or "text" in ctype or "xml" in ctype:
            self._parse_html(pg, raw, r.encoding)
        else:
            pg.blocked = f"unsupported content-type {ctype or 'unknown'}"
        return self._done(pg, purpose, r.status_code, True, len(raw))

    def _parse_pdf(self, pg, raw):
        try:
            from pypdf import PdfReader
            rd = PdfReader(io.BytesIO(raw))
            pg.text = "\n".join((p.extract_text() or "") for p in rd.pages[:40])
            pg.ok = bool(pg.text.strip())
            if not pg.ok:
                pg.blocked = "PDF has no extractable text (scanned?) - OCR not attempted"
        except Exception as e:
            pg.blocked = f"PDF unreadable: {type(e).__name__}"

    def _parse_html(self, pg, raw, enc):
        soup = BeautifulSoup(raw.decode(enc or "utf-8", errors="replace"), "lxml")
        pg.title = (soup.title.get_text(" ", strip=True) if soup.title else "")[:200]
        has_pw = bool(soup.find("input", {"type": "password"}))
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        for a in soup.find_all("a", href=True):
            h = a["href"].strip()
            if h.lower().startswith(("javascript:", "#")):
                continue
            pg.links.append((a.get_text(" ", strip=True)[:120], urljoin(pg.url, h)))
        for tb in soup.find_all("table"):
            rows = []
            for tr in tb.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(cells)
            if len(rows) >= 2:
                pg.tables.append(rows)
        pg.text = soup.get_text(" ", strip=True)
        if (pg.status in (403, 503) or len(pg.text) < 3000) and BLOCK_RE.search(pg.text):
            pg.blocked = "CAPTCHA / bot-protection page (not bypassed)"
        elif has_pw and len(pg.text) < 1500:
            pg.blocked = "login wall (not bypassed)"
        else:
            pg.ok = True

    def _done(self, pg, purpose, status, allowed, nbytes):
        note = pg.blocked or "ok"
        self.db.visit(self.run_id, pg.url, status, allowed, nbytes, note, purpose)
        if pg.ok:
            self.db.save_page(pg.url, pg.status, pg.title, pg.text)
        return pg

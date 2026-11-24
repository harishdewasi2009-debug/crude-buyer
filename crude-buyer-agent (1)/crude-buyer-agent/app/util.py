"""Small shared helpers: text normalisation, regex extractors, URL helpers."""
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

SUFFIX = {"pvt", "private", "ltd", "limited", "llp", "inc", "corp", "corporation", "co", "company", "the", "plc"}

CIN_RE = re.compile(r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:tel|telephone|phone|ph|call|board|fax)\b[^\d+]{0,15}(\+?\d[\d\s\-()/]{7,18}\d)", re.I)
FREE_MAIL = {"gmail.com", "yahoo.com", "yahoo.in", "hotmail.com", "outlook.com", "rediffmail.com", "proton.me",
             "protonmail.com", "aol.com", "live.com", "icloud.com", "mail.com"}

INDIAN_STATES = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
                 "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
                 "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
                 "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Chandigarh",
                 "Puducherry", "Jammu and Kashmir", "Ladakh"]


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def name_key(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower().replace("&", " and "))
    return " ".join(t for t in s.split() if t not in SUFFIX)


def domain_of(url: str) -> str:
    try:
        h = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return h[4:] if h.startswith("www.") else h


def base_domain(url_or_host: str) -> str:
    h = domain_of(url_or_host) if "/" in url_or_host else url_or_host.lower()
    labels = h.split(".")
    if len(labels) >= 3 and labels[-2] in {"co", "com", "org", "net", "gov", "nic", "ac", "edu"} and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else h


def same_site(a: str, b: str) -> bool:
    return base_domain(a) == base_domain(b)


def norm_text(s: str) -> str:
    s = (s or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_in_text(quote: str, text: str) -> bool:
    """True only if the quote really appears in the page text (anti-hallucination gate)."""
    q, t = norm_text(quote), norm_text(text)
    if len(q) < 12 or not t:
        return False
    if q in t:
        return True
    parts = [p.strip() for p in re.split(r"\.\.\.|…", q) if len(p.strip()) >= 15]
    return bool(parts) and all(p in t for p in parts) and len(parts) > 1


def sentences(text: str):
    for s in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        s = s.strip()
        if 40 <= len(s) <= 420:
            yield s


def clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n]


def email_is_official(email: str, site_url: str) -> bool:
    ed = email.split("@")[-1].lower()
    sd = base_domain(site_url)
    return bool(sd) and (ed == sd or ed.endswith("." + sd))


def subject_matches(subject: str, company_name: str) -> bool:
    a, b = set(name_key(subject).split()), set(name_key(company_name).split())
    return bool(a and b and (a & b))

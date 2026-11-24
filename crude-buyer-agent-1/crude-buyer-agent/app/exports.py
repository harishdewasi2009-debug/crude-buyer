"""Report generation: xlsx, csv, pdf. Every material claim carries a source URL."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .util import now_iso

NF = "NOT FOUND"
LABEL = {"A": "A — CONFIRMED BUYER", "B": "B — POTENTIAL BUYER", "C": "C — REFINERY-LINKED VENDOR",
         "D": "D — NOT A BUYER", "UNCLASSIFIED": "UNCLASSIFIED"}

COLS = [("Company name", "name"), ("Country", "country"), ("State", "state"), ("City", "city"),
        ("Company category", "category"), ("Ownership", "ownership"), ("CIN", "cin"), ("Website", "website"),
        ("Refinery connection", "refinery_connection"), ("Crude oil purchasing evidence", "purchase_evidence"),
        ("Procurement contact", "procurement_contact"), ("Procurement email", "procurement_email"),
        ("Official email", "official_email"), ("Phone number", "phone"), ("LinkedIn", "linkedin"),
        ("Tender reference", "tender_reference"), ("Award date", "award_date"),
        ("Supplier registration link", "supplier_reg_link"), ("Buyer classification", "classification"),
        ("Buyer type", "buyer_type"), ("Confidence", "confidence"), ("Verification status", "verification_status"),
        ("Evidence URLs", "evidence_urls"), ("Last research date", "last_research_date"),
        ("Follow-up status", "follow_up_status"), ("Notes", "notes")]


def safe(v):
    """Neutralise spreadsheet-formula injection from scraped text and show NOT FOUND for blanks."""
    if v is None or v == "":
        return NF
    if isinstance(v, (int, float)):
        return v
    s = str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


def cell(row, key):
    v = row.get(key)
    if key == "classification":
        return LABEL.get(v, v)
    if key == "evidence_urls":
        try:
            v = "\n".join(json.loads(v or "[]"))
        except Exception:
            pass
    return safe(v)


def rows_for(db, classes=None):
    sql = "SELECT * FROM companies"
    args = ()
    if classes:
        sql += f" WHERE classification IN ({','.join('?' * len(classes))})"
        args = tuple(classes)
    return db.q(sql + " ORDER BY classification, name", args)


def write_csv(db, path, classes):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([h for h, _ in COLS])
        for r in rows_for(db, classes):
            w.writerow([cell(r, k) for _, k in COLS])


def _sheet(ws, header, data, widths=None):
    ws.append(header)
    for r in data:
        ws.append(r)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F3A5F")
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, h in enumerate(header, 1):
        ws.column_dimensions[get_column_letter(i)].width = (widths or {}).get(h, min(max(len(h) + 4, 14), 40))
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions


def write_master(db, path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Primary Prospects (A+B)"
    hdr = [h for h, _ in COLS]
    _sheet(ws, hdr, [[cell(r, k) for _, k in COLS] for r in rows_for(db, ["A", "B"])])
    ws2 = wb.create_sheet("All Companies")
    _sheet(ws2, hdr, [[cell(r, k) for _, k in COLS] for r in rows_for(db)])
    ws3 = wb.create_sheet("Tender Awards")
    t = db.q("SELECT * FROM tenders ORDER BY id")
    _sheet(ws3, ["Issuer", "Tender no", "Successful bidder", "Award date", "Description", "Product category",
                 "Value", "Duration", "Direction", "Record kind", "Quote verified", "Source URL"],
           [[safe(x["source_name"]), safe(x["tender_no"]), safe(x["bidder"]), safe(x["award_date"]), safe(x["description"]),
             safe(x["product_category"]), safe(x["contract_value"]), safe(x["duration"]), safe(x["direction"]),
             safe(x["kind"]), "yes" if x["verified"] else "no", safe(x["source_url"])] for x in t])
    ws4 = wb.create_sheet("Evidence")
    ev = db.q("SELECT e.*, c.name FROM evidence e JOIN companies c ON c.id=e.company_id ORDER BY c.name, e.verified DESC")
    _sheet(ws4, ["Company", "Evidence kind", "Quote (verbatim from page)", "Verified in page text", "Method",
                 "Evidence date", "Source URL"],
           [[safe(x["name"]), safe(x["kind"]), safe(x["quote"]), "yes" if x["verified"] else "no", safe(x["method"]),
             safe(x["evidence_date"]), safe(x["url"])] for x in ev], {"Quote (verbatim from page)": 70, "Source URL": 60})
    ws5 = wb.create_sheet("About this file")
    for line in [f"Generated: {now_iso()}", "Primary prospects = classes A and B only.",
                 "A = direct crude purchasing with verified, QC-checked evidence. B = relevant but purchasing authority not established.",
                 "Only evidence whose quote was found verbatim in a fetched page is marked verified.",
                 "Contacts are copied from public pages; NOT FOUND means none was located - nothing is guessed.",
                 "Run your own sanctions / KYC screening before any dealing. No offers are sent by this system."]:
        ws5.append([line])
    ws5.column_dimensions["A"].width = 120
    wb.save(path)


def write_contacts(db, path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Procurement Contacts"
    hdr = ["Company", "Class", "Procurement contact", "Procurement email", "Official email", "Phone", "LinkedIn company page",
           "LinkedIn profiles (as linked on company site)", "Supplier registration link", "Contact source URL",
           "Contact verified from page", "Website", "Follow-up status"]
    data = []
    for r in rows_for(db, ["A", "B"]):
        try:
            li = "\n".join(f"{p.get('label', '')} {p['url']}".strip() for p in json.loads(r.get("linkedin_people") or "[]"))
        except Exception:
            li = ""
        data.append([safe(r["name"]), LABEL.get(r["classification"]), safe(r["procurement_contact"]), safe(r["procurement_email"]),
                     safe(r["official_email"]), safe(r["phone"]), safe(r["linkedin"]), safe(li), safe(r["supplier_reg_link"]),
                     safe(r["contact_source_url"]), "yes" if r["contacts_verified"] else "no", safe(r["website"]),
                     safe(r["follow_up_status"])])
    _sheet(ws, hdr, data)
    wb.save(path)


def write_dd_pdf(db, path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib import colors
    ss = getSampleStyleSheet()
    P = lambda t, st="BodyText": Paragraph(escape(str(t)), ss[st])
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Company Due Diligence Reports")
    story = [P("Company Due Diligence Reports", "Title"), P(f"Generated {now_iso()} · classes A and B"),
             P("Evidence is shown with its source URL. A claim is 'verified' only if its quote was found verbatim in the fetched page."),
             Spacer(1, 8)]
    comps = rows_for(db, ["A", "B"])
    if not comps:
        story.append(P("No class A or B companies have been verified yet."))
    for c in comps:
        story += [PageBreak(), P(c["name"], "Heading1")]
        tbl = [["Classification", LABEL.get(c["classification"])], ["Confidence", c.get("confidence") or NF],
               ["Verification status", c.get("verification_status") or NF], ["Buyer type", c.get("buyer_type") or NF],
               ["Ownership", c.get("ownership") or NF], ["CIN", c.get("cin") or NF], ["Website", c.get("website") or NF],
               ["Refinery connection", c.get("refinery_connection") or NF], ["Procurement contact", c.get("procurement_contact") or NF],
               ["Procurement email", c.get("procurement_email") or NF], ["Phone", c.get("phone") or NF]]
        t = Table([[P(a, "BodyText"), P(b, "BodyText")] for a, b in tbl], colWidths=[45 * mm, 120 * mm])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story += [t, Spacer(1, 6), P("Evidence", "Heading2")]
        evs = db.evidence_for(c["id"])
        if not evs:
            story.append(P(NF))
        for e in evs[:15]:
            story += [P(f"[{e['kind']}] {'VERIFIED' if e['verified'] else 'UNVERIFIED'} · {e['method']} · {e['evidence_date'] or 'date n/a'}"),
                      P(f"“{e['quote']}”", "Italic"), P(f"Source: {e['url']}"), Spacer(1, 3)]
        story.append(P("Red flags / due diligence", "Heading2"))
        flags = db.q("SELECT * FROM dd_flags WHERE company_id=? ORDER BY severity", (c["id"],))
        if not flags:
            story.append(P("None raised by automated checks (this is not a clearance)."))
        for f in flags:
            story.append(P(f"[{f['severity'].upper()}] {f['code']}: {f['text']}" + (f" Source: {f['source_url']}" if f["source_url"] else "")))
        if c.get("notes"):
            story += [P("Notes", "Heading2"), P(c["notes"])]
    doc.build(story)


EXPORTS = {
    "Indian Crude Oil Buyer Master List.xlsx": lambda db, p: write_master(db, p),
    "Confirmed Buyers.csv": lambda db, p: write_csv(db, p, ["A"]),
    "Potential Buyers.csv": lambda db, p: write_csv(db, p, ["B"]),
    "Refinery-Linked Vendors.csv": lambda db, p: write_csv(db, p, ["C"]),
    "Procurement Contacts.xlsx": lambda db, p: write_contacts(db, p),
    "Company Due Diligence Reports.pdf": lambda db, p: write_dd_pdf(db, p),
}


def generate(db, out_dir, name=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for fn, fnc in EXPORTS.items():
        if name and fn != name:
            continue
        fnc(db, out_dir / fn)
        made.append(fn)
    return made

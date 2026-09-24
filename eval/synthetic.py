"""Synthetic business documents with exact ground truth.

Public born-digital PDFs cover government forms and papers, but not the bank
statements, pay stubs, letters and invoices that dominate real scanned traffic.
These are generated with PyMuPDF so their text layer is exactly what was drawn.
All names, numbers and addresses are fabricated from fixed word lists with a
fixed seed: no real personal data, and every build is identical.
"""

import random

import fitz

FIRST = ["Maria", "James", "Aisha", "Daniel", "Linh", "Robert", "Sofia", "Kwame", "Elena", "Marcus"]
LAST = ["Alvarez", "Brennan", "Okafor", "Lindqvist", "Nguyen", "Castillo", "Whitaker", "Haddad", "Petrov", "Morales"]
STREETS = ["Maple Ave", "Harbor Rd", "Cedar Lane", "Lincoln Blvd", "Ridge St", "Orchard Way"]
CITIES = [("Springfield", "IL", "62704"), ("Fairview", "OR", "97024"), ("Riverton", "WY", "82501"),
          ("Madison", "WI", "53703"), ("Greenville", "SC", "29601")]
MERCHANTS = ["GROCERY OUTLET #214", "SHELL OIL 57442", "CITY WATER UTILITY", "PAYROLL DEPOSIT ACME CORP",
             "AMAZON MKTPLACE", "WALGREENS #0931", "RENT PAYMENT ONLINE", "ATM WITHDRAWAL 1180",
             "VERIZON WIRELESS", "TRANSFER TO SAVINGS", "COSTCO WHSE #447", "DOCTOR COPAY CLINIC"]

W, H = 612, 792          # US Letter in points
M = 54                   # margin


def _person(rng):
    city, state, zip_ = rng.choice(CITIES)
    return (f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            f"{rng.randint(100, 9899)} {rng.choice(STREETS)}",
            f"{city}, {state} {zip_}")


def _text(page, x, y, s, size=10, font="helv"):
    page.insert_text((x, y), s, fontsize=size, fontname=font)


def _rule(page, y, x0=M, x1=W - M, width=0.6):
    page.draw_line((x0, y), (x1, y), width=width)


def bank_statement(rng) -> fitz.Document:
    doc = fitz.open()
    name, street, cityline = _person(rng)
    balance = rng.uniform(1200, 5200)
    for pno in range(2):
        p = doc.new_page(width=W, height=H)
        _text(p, M, 60, "FIRST CAPITAL COMMUNITY BANK", 15, "hebo")
        _text(p, M, 76, "PO Box 4410, Columbus, OH 43216  |  1-800-555-0142", 8.5)
        _text(p, W - 200, 60, f"Page {pno + 1} of 2", 9)
        _text(p, M, 108, name.upper(), 10)
        _text(p, M, 121, street.upper(), 10)
        _text(p, M, 134, cityline.upper(), 10)
        _text(p, W - 250, 108, "Statement Period: 07/01/2026 - 07/31/2026", 9)
        _text(p, W - 250, 121, f"Account Number: XXXX-XXXX-{rng.randint(1000, 9999)}", 9)
        _text(p, W - 250, 134, "Everyday Checking", 9)
        y = 170
        if pno == 0:
            p.draw_rect(fitz.Rect(M, y, W - M, y + 70), width=0.8)
            _text(p, M + 10, y + 18, "ACCOUNT SUMMARY", 10, "hebo")
            _text(p, M + 10, y + 36, f"Beginning Balance   ${balance:,.2f}", 9.5)
            _text(p, M + 10, y + 52, f"Deposits and Credits   ${rng.uniform(2500, 4800):,.2f}", 9.5)
            _text(p, 320, y + 36, f"Withdrawals and Debits   ${rng.uniform(1800, 4200):,.2f}", 9.5)
            _text(p, 320, y + 52, f"Service Fees   ${rng.choice([0, 5, 12]):,.2f}", 9.5)
            y += 100
        _text(p, M, y, "DATE", 9, "hebo"); _text(p, M + 60, y, "DESCRIPTION", 9, "hebo")
        _text(p, 380, y, "AMOUNT", 9, "hebo"); _text(p, 470, y, "BALANCE", 9, "hebo")
        _rule(p, y + 5)
        y += 20
        while y < H - 80:
            amt = rng.choice([-1, -1, -1, 1]) * rng.uniform(4, 900)
            balance += amt
            _text(p, M, y, f"07/{rng.randint(1, 31):02d}", 9)
            _text(p, M + 60, y, rng.choice(MERCHANTS), 9)
            _text(p, 380, y, f"{amt:,.2f}", 9, "cour")
            _text(p, 470, y, f"{balance:,.2f}", 9, "cour")
            y += 17
        _text(p, M, H - 40, "Member FDIC. Questions about your statement? Call the number above.", 7.5)
    return doc


def pay_stub(rng) -> fitz.Document:
    doc = fitz.open(); p = doc.new_page(width=W, height=H)
    name, street, cityline = _person(rng)
    _text(p, M, 64, "NORTHWIND LOGISTICS LLC", 14, "hebo")
    _text(p, M, 80, "2250 Industrial Pkwy, Dayton, OH 45414", 9)
    _text(p, W - 190, 64, "EARNINGS STATEMENT", 11, "hebo")
    _text(p, W - 190, 80, f"Check Date: 08/{rng.randint(1, 28):02d}/2026", 9)
    _text(p, M, 120, f"Employee: {name}", 10); _text(p, M, 134, street, 10); _text(p, M, 148, cityline, 10)
    _text(p, 330, 120, f"Employee ID: {rng.randint(10000, 99999)}", 9.5)
    _text(p, 330, 134, "Pay Period: 07/16/2026 - 07/31/2026", 9.5)
    _text(p, 330, 148, "Pay Frequency: Semi-Monthly", 9.5)
    y = 190
    rate = rng.uniform(17, 42); hours = rng.choice([80, 86.5, 72, 80])
    rows = [("Regular", f"{rate:.2f}", f"{hours:.2f}", rate * hours),
            ("Overtime", f"{rate * 1.5:.2f}", f"{rng.choice([0, 4, 6.5]):.2f}", rate * 1.5 * 4),
            ("Holiday", f"{rate:.2f}", "8.00", rate * 8)]
    _text(p, M, y, "EARNINGS", 10, "hebo"); _text(p, 200, y, "RATE", 9, "hebo")
    _text(p, 280, y, "HOURS", 9, "hebo"); _text(p, 360, y, "CURRENT", 9, "hebo"); _text(p, 450, y, "YEAR TO DATE", 9, "hebo")
    _rule(p, y + 5); y += 20
    gross = 0
    for label, r, h, cur in rows:
        gross += cur
        _text(p, M, y, label, 9.5); _text(p, 200, y, r, 9.5, "cour"); _text(p, 280, y, h, 9.5, "cour")
        _text(p, 360, y, f"{cur:,.2f}", 9.5, "cour"); _text(p, 450, y, f"{cur * 14:,.2f}", 9.5, "cour")
        y += 16
    y += 20
    _text(p, M, y, "DEDUCTIONS", 10, "hebo"); _rule(p, y + 5); y += 20
    for label, frac in [("Federal Income Tax", .11), ("Social Security", .062), ("Medicare", .0145),
                        ("State Income Tax", .045), ("401(k) Contribution", .05), ("Dental Plan", .008)]:
        _text(p, M, y, label, 9.5); _text(p, 360, y, f"{gross * frac:,.2f}", 9.5, "cour")
        _text(p, 450, y, f"{gross * frac * 14:,.2f}", 9.5, "cour"); y += 16
    y += 24
    _text(p, M, y, f"GROSS PAY {gross:,.2f}", 11, "hebo")
    _text(p, 330, y, f"NET PAY {gross * 0.7155:,.2f}", 11, "hebo")
    _text(p, M, y + 40, "Direct deposit to account ending 4471. This is not a check.", 8.5)
    return doc


def letter(rng) -> fitz.Document:
    doc = fitz.open(); p = doc.new_page(width=W, height=H)
    name, street, cityline = _person(rng)
    _text(p, M, 70, "Greenfield Property Management", 14, "tibo")
    _text(p, M, 86, "88 Commerce Street, Suite 300, Albany, NY 12207", 9, "tiro")
    _text(p, M, 130, "September 3, 2026", 11, "tiro")
    _text(p, M, 165, name, 11, "tiro"); _text(p, M, 179, street, 11, "tiro"); _text(p, M, 193, cityline, 11, "tiro")
    _text(p, M, 230, f"Dear {name.split()[0]},", 11, "tiro")
    body = ("We are writing to confirm the results of your annual recertification for the apartment "
            "you currently occupy. Based on the income and household information you provided, your "
            "monthly tenant rent will change effective November 1, 2026. Please review the enclosed "
            "certification carefully and contact our office within fourteen days if any information "
            "is incomplete or incorrect. A copy of your signed lease addendum must be returned before "
            "the effective date. If you need assistance completing the forms, our staff are available "
            "Monday through Friday between nine in the morning and five in the afternoon. Thank you for "
            "your continued cooperation and for being a valued resident of our community.")
    rect = fitz.Rect(M, 245, W - M, 520)
    p.insert_textbox(rect, body, fontsize=11, fontname="tiro", lineheight=1.35)
    _text(p, M, 560, "Sincerely,", 11, "tiro")
    _text(p, M, 610, "Patricia Holloway", 11, "tiro"); _text(p, M, 624, "Compliance Manager", 11, "tiro")
    return doc


def invoice(rng) -> fitz.Document:
    doc = fitz.open(); p = doc.new_page(width=W, height=H)
    name, street, cityline = _person(rng)
    _text(p, M, 64, "BRIGHTLINE ELECTRICAL SUPPLY", 14, "hebo")
    _text(p, W - 170, 64, "INVOICE", 18, "hebo")
    _text(p, W - 170, 84, f"Invoice # {rng.randint(100000, 999999)}", 9.5)
    _text(p, W - 170, 98, "Date: 08/14/2026", 9.5); _text(p, W - 170, 112, "Terms: Net 30", 9.5)
    _text(p, M, 120, "BILL TO", 9, "hebo"); _text(p, M, 134, name, 10); _text(p, M, 148, street, 10); _text(p, M, 162, cityline, 10)
    y = 200
    _text(p, M, y, "QTY", 9, "hebo"); _text(p, 100, y, "DESCRIPTION", 9, "hebo")
    _text(p, 380, y, "UNIT PRICE", 9, "hebo"); _text(p, 470, y, "AMOUNT", 9, "hebo"); _rule(p, y + 5); y += 20
    items = ["12/2 NM-B Romex Cable 250ft", "20A Single Pole Breaker", "Duplex Receptacle Tamper Resistant",
             "LED Recessed Downlight 6in", "Junction Box 4in Square", "Wire Nuts Assorted 100pk",
             "GFCI Outlet 20A White", "PVC Conduit 3/4in 10ft", "Smart Dimmer Switch", "Grounding Rod 8ft"]
    total = 0
    for it in items:
        q = rng.randint(1, 24); u = rng.uniform(1.5, 180); total += q * u
        _text(p, M, y, str(q), 9.5); _text(p, 100, y, it, 9.5)
        _text(p, 380, y, f"{u:,.2f}", 9.5, "cour"); _text(p, 470, y, f"{q * u:,.2f}", 9.5, "cour"); y += 17
    _rule(p, y); y += 20
    _text(p, 380, y, "Subtotal", 9.5); _text(p, 470, y, f"{total:,.2f}", 9.5, "cour")
    _text(p, 380, y + 16, "Sales Tax 7.25%", 9.5); _text(p, 470, y + 16, f"{total * .0725:,.2f}", 9.5, "cour")
    _text(p, 380, y + 34, "TOTAL DUE", 10, "hebo"); _text(p, 470, y + 34, f"{total * 1.0725:,.2f}", 10, "cour")
    return doc


def dense_table(rng) -> fitz.Document:
    doc = fitz.open(); p = doc.new_page(width=W, height=H)
    _text(p, M, 56, "Table 4. Monthly Utility Consumption by Building, Fiscal Year 2026", 10, "hebo")
    cols = ["Bldg", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Total"]
    xs = [M + i * 63 for i in range(len(cols))]
    y = 80
    for x, c in zip(xs, cols): _text(p, x, y, c, 7.5, "hebo")
    _rule(p, y + 4); y += 14
    for r in range(48):
        vals = [rng.randint(1200, 9800) for _ in range(6)]
        cells = [f"B-{r + 101}"] + [f"{v:,}" for v in vals] + [f"{sum(vals):,}"]
        for x, c in zip(xs, cells): _text(p, x, y, c, 7.5, "cour")
        y += 13.2
    return doc


def cover_page(rng) -> fitz.Document:
    """Sparse page: the service's content-area check skips pages like this."""
    doc = fitz.open(); p = doc.new_page(width=W, height=H)
    _text(p, 150, 330, "Annual Compliance Report", 24, "hebo")
    _text(p, 230, 365, "Fiscal Year 2026", 14, "helv")
    return doc


def signature_page(rng) -> fitz.Document:
    """Sparse closing page: a few lines at the top, the rest blank."""
    doc = fitz.open(); p = doc.new_page(width=W, height=H)
    _text(p, M, 80, "IN WITNESS WHEREOF, the parties have executed this agreement.", 10.5, "tiro")
    _text(p, M, 130, "Tenant Signature: ______________________   Date: __________", 10.5, "tiro")
    _text(p, M, 170, "Landlord Signature: ____________________   Date: __________", 10.5, "tiro")
    return doc


BUILDERS = {
    "syn_bank_statement": ("statement", bank_statement),
    "syn_pay_stub": ("paystub", pay_stub),
    "syn_letter": ("letter", letter),
    "syn_invoice": ("invoice", invoice),
    "syn_dense_table": ("dense_table", dense_table),
    "syn_cover_page": ("sparse", cover_page),
    "syn_signature_page": ("sparse", signature_page),
}


def build_all(out_dir: str, seed: int = 20260924) -> dict:
    """Write every synthetic PDF to out_dir; return {name: category}."""
    made = {}
    for i, (name, (category, fn)) in enumerate(BUILDERS.items()):
        doc = fn(random.Random(seed + i))
        doc.save(f"{out_dir}/{name}.pdf")
        made[name] = category
    return made

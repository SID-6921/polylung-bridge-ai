# -*- coding: utf-8 -*-
import re, json, subprocess, sys
from pathlib import Path

pdf_path = Path(r"C:\Users\nanda\Downloads\1-s2.0-S0753332225003166-main.pdf")
if not pdf_path.exists():
    print("ERROR: PDF_NOT_FOUND")
    sys.exit(1)

try:
    import fitz
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf", "-q"])
    import fitz

cytokines = {
    "TNF-a": [r"tnf\s*[-]?\s*(?:a|alpha|α)"],
    "IL-1b": [r"il\s*[-]?\s*1\s*(?:b|beta|β)"],
    "IL-5": [r"il\s*[-]?\s*5\b"],
    "IL-6": [r"il\s*[-]?\s*6\b"],
    "MIP-2a": [r"mip\s*[-]?\s*2\s*(?:a|alpha|α)"]
}

num_unit_re = re.compile(r"(?P<value>\b\d{1,4}(?:\.\d+)?(?:\s*[±]\s*\d{1,4}(?:\.\d+)?)?)\s*(?P<unit>pg\s*/\s*mL|ng\s*/\s*mL|pg/mL|ng/mL|pg\s*mL\^-?1|ng\s*mL\^-?1|pg\s*mL-1|ng\s*mL-1)\b", re.I)
num_pm_re = re.compile(r"\b\d{1,4}(?:\.\d+)?\s*[±]\s*\d{1,4}(?:\.\d+)?\b")
group_re = re.compile(r"\b(control|normal|model|lps|bleomycin|blm|vehicle|sham|psmps|treated|low-dose|high-dose|group\s*[A-Za-z0-9]+)\b", re.I)
strip_labels_re = re.compile(r"tnf\s*[-]?\s*(?:a|alpha|α)|il\s*[-]?\s*1\s*(?:b|beta|β)|il\s*[-]?\s*5\b|il\s*[-]?\s*6\b|mip\s*[-]?\s*2\s*(?:a|alpha|α)", re.I)

rows, mentions = [], []

doc = fitz.open(str(pdf_path))
for pidx, page in enumerate(doc, start=1):
    text = page.get_text("text") or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        low = line.lower()
        for ck, pats in cytokines.items():
            if any(re.search(p, low, re.I) for p in pats):
                window_lines = lines[max(0, i-3):min(len(lines), i+4)]
                window = " | ".join(window_lines)
                if not re.search(r"\bbal\b|bronchoalveolar", window, re.I):
                    continue
                sanitized = strip_labels_re.sub("", window)
                found = False
                for m in num_unit_re.finditer(sanitized):
                    found = True
                    g = group_re.search(window)
                    rows.append({
                        "cytokine": ck,
                        "group": g.group(0) if g else "",
                        "value": m.group("value"),
                        "unit": re.sub(r"\s+", "", m.group("unit")),
                        "source_page": pidx,
                        "source_line": i+1,
                        "excerpt": window[:260]
                    })
                if not found:
                    # keep only contextual mention; ignore bare integers without units to avoid label-number artifacts
                    mentions.append({
                        "cytokine": ck,
                        "source_page": pidx,
                        "source_line": i+1,
                        "excerpt": window[:260]
                    })

# dedupe
seen = set(); uniq=[]
for r in rows:
    k=(r['cytokine'],r['group'],r['value'],r['unit'],r['source_page'],r['source_line'])
    if k not in seen:
        seen.add(k); uniq.append(r)

seen_m=set(); uniq_m=[]
for r in mentions:
    k=(r['cytokine'],r['source_page'],r['source_line'])
    if k not in seen_m:
        seen_m.add(k); uniq_m.append(r)

print("===EXTRACT_START===")
if uniq:
    print(json.dumps(uniq, ensure_ascii=False, indent=2))
else:
    print(json.dumps({"explicit_numeric_values_found": False, "reason": "No BAL-context cytokine values with explicit units/± patterns were found in extractable PDF text.", "mentions": uniq_m[:20]}, ensure_ascii=False, indent=2))
print("===EXTRACT_END===")

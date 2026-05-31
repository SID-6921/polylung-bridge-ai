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

num_re = re.compile(r"(?P<value>\b\d+(?:\.\d+)?(?:\s*[±+-]\s*\d+(?:\.\d+)?)?)\s*(?P<unit>pg\s*/\s*mL|pg/mL|ng\s*/\s*mL|ng/mL|pg\s*mL\^-?1|ng\s*mL\^-?1)?", re.I)
group_re = re.compile(r"\b(control|normal|model|lps|bleomycin|blm|vehicle|sham|treated|group\s*[A-Za-z0-9]+)\b", re.I)

rows = []

doc = fitz.open(str(pdf_path))
for pidx, page in enumerate(doc, start=1):
    text = page.get_text("text") or ""
    lines = [ln.strip() for ln in text.splitlines()]
    for i, line in enumerate(lines):
        low = line.lower()
        for ck, pats in cytokines.items():
            if any(re.search(p, low, re.I) for p in pats):
                context = " ".join(lines[max(0, i-2):i+3])
                if not re.search(r"\bBAL\b|bronchoalveolar", context, re.I):
                    continue
                window_lines = lines[max(0, i-2):min(len(lines), i+3)]
                window = " | ".join(window_lines)
                nums = list(num_re.finditer(window))
                if nums:
                    for m in nums:
                        val = m.group("value")
                        unit = m.group("unit") or ""
                        if not re.search(r"\d", val):
                            continue
                        g = group_re.search(window)
                        group = g.group(0) if g else ""
                        rows.append({
                            "cytokine": ck,
                            "group": group,
                            "value": val.strip(),
                            "unit": re.sub(r"\s+", "", unit),
                            "source_page": pidx,
                            "source_line": i+1,
                            "excerpt": window[:260]
                        })
                else:
                    rows.append({
                        "cytokine": ck,
                        "group": "",
                        "value": "",
                        "unit": "",
                        "source_page": pidx,
                        "source_line": i+1,
                        "excerpt": window[:260]
                    })

seen = set()
uniq = []
for r in rows:
    k = (r["cytokine"], r["group"], r["value"], r["unit"], r["source_page"], r["source_line"])
    if k not in seen:
        seen.add(k)
        uniq.append(r)

explicit = [r for r in uniq if r["value"]]

print("===EXTRACT_START===")
if explicit:
    print(json.dumps(explicit, ensure_ascii=False, indent=2))
else:
    mentions = [r for r in uniq if not r["value"]]
    print(json.dumps({"explicit_numeric_values_found": False, "mentions": mentions[:20]}, ensure_ascii=False, indent=2))
print("===EXTRACT_END===")

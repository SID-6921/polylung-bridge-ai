import pandas as pd, glob, os, re

BASE = os.path.expanduser("~/polylung-bridge-ai/data/microplastics_datasets/unicamp_estuary_sem_ftir")
xlsx = os.path.join(BASE, "Dataset.xlsx")
df = pd.read_excel(xlsx)
print("Columns:", list(df.columns))
print("Rows:", len(df))

# Normalize Sample_ID and Polymer columns
df2 = df[df['Polymer'].notna()].copy()
print("Rows with confirmed Polymer:", len(df2))
print(df2['Polymer'].value_counts())

ftir_files = glob.glob(os.path.join(BASE, "FTIR", "FTIR", "RC_MP_*.csv"))
sem_files = glob.glob(os.path.join(BASE, "SEM-EDS", "SEM-EDS", "RC_MP_*.pdf"))

def sample_key_from_filename(fn, exts):
    base = os.path.basename(fn)
    for ext in exts:
        base = base.replace(ext, "")
    # strip trailing a/b suffix for SEM
    m = re.match(r'(RC_MP_\d+)', base)
    return m.group(1) if m else base

ftir_keys = set(sample_key_from_filename(f, ['.csv']) for f in ftir_files)
sem_keys = set(sample_key_from_filename(f, ['.pdf']) for f in sem_files)

print("\nFTIR sample keys found:", len(ftir_keys))
print("SEM sample keys found:", len(sem_keys))

df2['Sample_ID_str'] = df2['Sample_ID'].astype(str).str.strip()

results = []
for _, row in df2.iterrows():
    sid = row['Sample_ID_str']
    has_ftir = sid in ftir_keys
    has_sem = sid in sem_keys
    if has_ftir and has_sem:
        results.append((sid, row['Polymer']))

print("\n=== Particles with confirmed Polymer + SEM-EDS PDF + FTIR CSV ===")
print("n =", len(results))
from collections import Counter
c = Counter([p for _, p in results])
print(c)
for sid, p in sorted(results):
    print(sid, p)

# also list matching SEM files per sample id (for split a/b decisions)
print("\n=== SEM files per matched sample ===")
for sid, p in sorted(results):
    matches = sorted([f for f in sem_files if sample_key_from_filename(f, ['.pdf']) == sid])
    print(sid, p, [os.path.basename(m) for m in matches])

print("\n=== FTIR files per matched sample ===")
for sid, p in sorted(results):
    matches = sorted([f for f in ftir_files if sample_key_from_filename(f, ['.csv']) == sid])
    print(sid, p, [os.path.basename(m) for m in matches])

# PD-2 PSPII Weight Derivation

Files:
- `balkrishna_2024_extracted.csv`: cytokine values (TNFa, IL1b, IL5, IL6, MIP2a) from WebPlotDigitizer
- `pspii_weights_final.json`: normalized PSPII outputs

Generate weights:
```bash
python scripts/derive_pspii_weights.py --input evidence/public/pd2/balkrishna_2024_extracted.csv --output evidence/public/pd2/pspii_weights_final.json
```

Then sync to API runtime file:
```bash
copy evidence\public\pd2\pspii_weights_final.json data\pspii_weights_final.json
```

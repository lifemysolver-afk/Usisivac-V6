"""
Analizira Loptica.ipynb i izvlači sve relevantne informacije:
- Sve code ćelije
- Sve markdown ćelije (objašnjenja)
- Biblioteke koje se koriste
- Modeli i tehnike
- Metrike
"""
import json, re, os
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
nb_path = BASE / "data" / "Loptica.ipynb"

if not nb_path.exists():
    print(f"Greška: Fajl {nb_path} ne postoji.")
    exit(1)

with open(nb_path) as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Ukupno ćelija: {len(cells)}")

code_cells = [c for c in cells if c["cell_type"] == "code"]
md_cells   = [c for c in cells if c["cell_type"] == "markdown"]
print(f"  Code: {len(code_cells)}, Markdown: {len(md_cells)}")

# Izvuci sav kod
all_code = []
for i, c in enumerate(code_cells):
    src = "".join(c.get("source", []))
    all_code.append(src)

full_code = "\n\n# --- CELL ---\n\n".join(all_code)

# Izvuci sve markdowne
all_md = []
for c in md_cells:
    src = "".join(c.get("source", []))
    all_md.append(src)

# Pronađi imports
imports = []
for line in full_code.split("\n"):
    if line.strip().startswith("import ") or line.strip().startswith("from "):
        imports.append(line.strip())

print("\n=== IMPORTS ===")
for imp in sorted(set(imports)):
    print(f"  {imp}")

# Pronađi modele
model_patterns = [
    r'(XGBClassifier|XGBRegressor|xgb\.)',
    r'(LGBMClassifier|LGBMRegressor|lgb\.)',
    r'(CatBoostClassifier|CatBoostRegressor|cb\.)',
    r'(RandomForestClassifier|RandomForestRegressor)',
    r'(LogisticRegression|Ridge|Lasso)',
    r'(Sequential|keras|torch\.nn)',
]

found_models = []
for pattern in model_patterns:
    found_models.extend(re.findall(pattern, full_code))

print("\n=== MODELS ===")
for mod, count in Counter(found_models).items():
    print(f"  {mod}: {count}")

# Sačuvaj rezultate
out_code = BASE / "data" / "loptica_extracted_code.py"
out_md = BASE / "data" / "loptica_extracted_md.md"
out_analysis = BASE / "data" / "loptica_analysis.json"

out_code.write_text(full_code)
out_md.write_text("\n\n---\n\n".join(all_md))

analysis = {
    "n_code": len(code_cells),
    "n_md": len(md_cells),
    "imports": sorted(list(set(imports))),
    "models": dict(Counter(found_models))
}
out_analysis.write_text(json.dumps(analysis, indent=2))
print(f"\nAnaliza sačuvana u {out_analysis}")

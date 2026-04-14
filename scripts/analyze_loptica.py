"""
Analizira Loptica.ipynb i izvlači sve relevantne informacije:
- Sve code ćelije
- Sve markdown ćelije (objašnjenja)
- Biblioteke koje se koriste
- Modeli i tehnike
- Metrike
"""
import json, re
from pathlib import Path
from collections import Counter

nb_path = Path("./data/Loptica.ipynb")
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
    r'(CatBoostClassifier|CatBoostRegressor)',
    r'(RandomForestClassifier|RandomForestRegressor)',
    r'(LogisticRegression)',
    r'(GradientBoostingClassifier|GradientBoostingRegressor)',
    r'(SVC|SVR)',
    r'(MLPClassifier|MLPRegressor)',
    r'(Sequential|Dense|LSTM|Conv|Transformer)',
    r'(BallTree|KDTree|NearestNeighbors)',
    r'(KMeans|DBSCAN|AgglomerativeClustering)',
    r'(PCA|UMAP|TSNE)',
    r'(Pipeline|ColumnTransformer|StandardScaler|MinMaxScaler)',
    r'(cross_val_score|StratifiedKFold|KFold)',
    r'(GridSearchCV|RandomizedSearchCV|optuna)',
]

print("\n=== MODELI I TEHNIKE ===")
found_models = set()
for pat in model_patterns:
    matches = re.findall(pat, full_code)
    if matches:
        for m in set(matches):
            found_models.add(m)
            print(f"  FOUND: {m}")

# Pronađi metrike
metric_patterns = [
    r'(roc_auc_score|auc|AUC)',
    r'(accuracy_score|accuracy)',
    r'(f1_score|f1)',
    r'(mean_squared_error|mse|rmse|RMSE)',
    r'(mean_absolute_error|mae|MAE)',
    r'(r2_score|R2)',
    r'(log_loss)',
    r'(precision_score|recall_score)',
]

print("\n=== METRIKE ===")
found_metrics = set()
for pat in metric_patterns:
    matches = re.findall(pat, full_code)
    if matches:
        for m in set(matches):
            found_metrics.add(m)
            print(f"  FOUND: {m}")

# Pronađi dataset info
print("\n=== DATASET CLUES ===")
for line in full_code.split("\n"):
    if any(kw in line.lower() for kw in ["read_csv", "read_excel", "load_dataset", "pd.read", "columns", "shape", "target", "label"]):
        print(f"  {line.strip()[:120]}")

# Pronađi neuronske mreže
print("\n=== NEURONSKE MREŽE ===")
for line in full_code.split("\n"):
    if any(kw in line.lower() for kw in ["dense", "lstm", "conv", "embedding", "dropout", "relu", "sigmoid", "softmax", "torch", "tensorflow", "keras"]):
        print(f"  {line.strip()[:120]}")

# Sačuvaj ceo kod u fajl
out_code = Path("./data/loptica_extracted_code.py")
out_code.write_text(full_code)
print(f"\n=== SAVED: {out_code} ({len(full_code)} chars) ===")

# Sačuvaj markdown
out_md = Path("./data/loptica_extracted_md.md")
out_md.write_text("\n\n---\n\n".join(all_md))
print(f"=== SAVED: {out_md} ===")

# Sačuvaj analizu
analysis = {
    "total_cells": len(cells),
    "code_cells": len(code_cells),
    "markdown_cells": len(md_cells),
    "imports": sorted(set(imports)),
    "models_found": sorted(found_models),
    "metrics_found": sorted(found_metrics),
}
out_analysis = Path("./data/loptica_analysis.json")
out_analysis.write_text(json.dumps(analysis, indent=2))
print(f"=== SAVED: {out_analysis} ===")

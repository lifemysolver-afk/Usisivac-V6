import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from agents.research_agent import run

def trigger_research():
    print("Starting ResearchAgent for S6E3...")
    
    # 1. Ingest universal knowledge first
    print("Ingesting universal knowledge...")
    run({"action": "ingest"})
    
    # 2. Research specific topics for S6E3
    queries = [
        "customer churn prediction best practices",
        "tabular data competition winning strategies",
        "handling class imbalance in churn prediction",
        "feature engineering for telecommunications churn",
        "blending XGBoost LightGBM CatBoost for AUC",
        "Yggdrasil Decision Forests for tabular data",
        "ByT5 for tabular data feature extraction",
        "hierarchical attention for customer behavior sequences",
        "low-resource machine translation techniques for categorical features"
    ]
    
    for q in queries:
        print(f"Researching: {q}")
        run({"action": "research", "query": q, "domain": "tabular"})
    
    # 3. Extract Golden Recipe for S6E3
    print("Extracting Golden Recipe for S6E3...")
    problem_desc = Path(BASE / "data/s6e3_problem_description.txt").read_text()
    run({"action": "golden_recipe", "problem": problem_desc})
    
    print("ResearchAgent task completed.")

if __name__ == "__main__":
    trigger_research()

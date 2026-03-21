"""Phase 1 pseudo-labeling probe for S6E3 telco churn.

This script does three things:
1) Runs a narrow RAG query set for pseudo-labeling / semi-supervised evidence.
2) Measures seed stability on a high-confidence unlabeled slice.
3) Optionally retrains on pseudo-labeled rows and writes a Kaggle submission.

The implementation is intentionally self-contained so it can run even if the
project's higher-level V12 wrapper is not exposed as a stable import.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TARGET_COL = "Churn"
ID_COL = "id"

RAG_QUERIES = [
    "pseudo labeling kaggle tabular competition winner",
    "pseudo labeling telco churn",
    "semi-supervised kaggle playground",
]


@dataclass
class SeedResult:
    seed: int
    oof_auc: float
    test_mean: float
    test_std: float


@dataclass
class ProbeSummary:
    timestamp_utc: str
    seeds: List[int]
    base_seed_results: List[SeedResult]
    avg_oof_auc: float
    high_confidence_rate: float
    unanimity_rate: float
    pseudo_label_rate: float
    pseudo_label_count: int
    retrain_ran: bool
    output_submission: Optional[str]
    output_pseudo_train: Optional[str]
    rag_status: Dict[str, Any]


def _import_research_agent():
    try:
        from agents.research_agent import run as research_run  # type: ignore

        return research_run
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[RAG] ResearchAgent unavailable: {exc}")
        return None


def run_rag_queries(queries: Sequence[str]) -> Dict[str, Any]:
    research_run = _import_research_agent()
    results: Dict[str, Any] = {"available": bool(research_run), "queries": []}

    if research_run is None:
        return results

    try:
        research_run({"action": "ingest"})
    except Exception as exc:  # pragma: no cover - defensive
        results["ingest_error"] = str(exc)

    for query in queries:
        try:
            payload = research_run({"action": "research", "query": query, "domain": "tabular"})
            top_hits = []
            for item in payload.get("results", [])[:3]:
                top_hits.append(
                    {
                        "score": float(item.get("_score", 0.0)),
                        "source": item.get("_source_collection", "?"),
                        "content": str(item.get("content", ""))[:240],
                    }
                )
            results["queries"].append(
                {
                    "query": query,
                    "status": payload.get("status", "UNKNOWN"),
                    "found": payload.get("total_found", 0),
                    "top_hits": top_hits,
                }
            )
            print(f"[RAG] {query}: found={payload.get('total_found', 0)}")
        except Exception as exc:  # pragma: no cover - defensive
            results["queries"].append({"query": query, "error": str(exc)})
            print(f"[RAG] {query}: error={exc}")

    report_path = REPORT_DIR / "pseudo_label_rag_summary.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def load_data(train_path: Path = TRAIN_PATH, test_path: Path = TEST_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if TARGET_COL not in train.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in {train_path}")
    if ID_COL not in train.columns or ID_COL not in test.columns:
        raise ValueError(f"Missing id column '{ID_COL}' in train/test data")

    target_encoder = LabelEncoder()
    y = pd.Series(target_encoder.fit_transform(train[TARGET_COL].astype(str)), index=train.index, name=TARGET_COL)

    X = train.drop(columns=[TARGET_COL, ID_COL])
    X_test = test.drop(columns=[ID_COL])
    return train, test, y, X, X_test


def _build_models(seed: int):
    models: List[Tuple[str, Any]] = []

    try:
        import xgboost as xgb  # type: ignore

        models.append(
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=2000,
                    learning_rate=0.03,
                    max_depth=6,
                    min_child_weight=1.0,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    reg_alpha=0.0,
                    objective="binary:logistic",
                    tree_method="hist",
                    eval_metric="auc",
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        )
    except Exception as exc:
        print(f"[MODEL] XGBoost unavailable: {exc}")

    try:
        import lightgbm as lgb  # type: ignore

        models.append(
            (
                "lgb",
                lgb.LGBMClassifier(
                    n_estimators=2500,
                    learning_rate=0.03,
                    num_leaves=31,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_samples=20,
                    reg_alpha=0.0,
                    reg_lambda=0.0,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                ),
            )
        )
    except Exception as exc:
        print(f"[MODEL] LightGBM unavailable: {exc}")

    try:
        import catboost as cb  # type: ignore

        models.append(
            (
                "cat",
                cb.CatBoostClassifier(
                    iterations=2500,
                    learning_rate=0.03,
                    depth=6,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=seed,
                    verbose=False,
                    od_type="Iter",
                    od_wait=100,
                ),
            )
        )
    except Exception as exc:
        print(f"[MODEL] CatBoost unavailable: {exc}")

    if not models:
        raise RuntimeError("No supported tree models are available in the environment")

    return models


def _prepare_fold_matrices(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    cat_cols = [c for c in X_tr.columns if X_tr[c].dtype == "object" or str(X_tr[c].dtype).startswith("category")]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]

    X_tr_num = X_tr[num_cols].copy()
    X_val_num = X_val[num_cols].copy()
    X_test_num = X_test[num_cols].copy()

    medians = X_tr_num.median(numeric_only=True)
    X_tr_num = X_tr_num.fillna(medians)
    X_val_num = X_val_num.fillna(medians)
    X_test_num = X_test_num.fillna(medians)

    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr_cat = encoder.fit_transform(X_tr[cat_cols].fillna("__missing__").astype(str))
        X_val_cat = encoder.transform(X_val[cat_cols].fillna("__missing__").astype(str))
        X_test_cat = encoder.transform(X_test[cat_cols].fillna("__missing__").astype(str))
        X_tr_out = np.hstack([X_tr_num.to_numpy(dtype=np.float32), X_tr_cat.astype(np.float32)])
        X_val_out = np.hstack([X_val_num.to_numpy(dtype=np.float32), X_val_cat.astype(np.float32)])
        X_test_out = np.hstack([X_test_num.to_numpy(dtype=np.float32), X_test_cat.astype(np.float32)])
    else:
        X_tr_out = X_tr_num.to_numpy(dtype=np.float32)
        X_val_out = X_val_num.to_numpy(dtype=np.float32)
        X_test_out = X_test_num.to_numpy(dtype=np.float32)

    return X_tr_out, X_val_out, X_test_out, cat_cols, num_cols


def _fit_one_model(name: str, model: Any, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    if name == "xgb":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "lgb":
        import lightgbm as lgb  # type: ignore

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    elif name == "cat":
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    else:
        model.fit(X_tr, y_tr)

    return model.predict_proba(X_val)[:, 1]


def _predict_one_model(name: str, model: Any, X_test: np.ndarray) -> np.ndarray:
    return model.predict_proba(X_test)[:, 1]


def run_seeded_cv(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr_df = X.iloc[tr_idx].copy()
        X_val_df = X.iloc[val_idx].copy()
        y_tr = y.iloc[tr_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        X_tr, X_val, X_te, _, _ = _prepare_fold_matrices(X_tr_df, X_val_df, X_test)
        fold_preds = np.zeros(len(X_val), dtype=np.float32)
        fold_test_preds = np.zeros(len(X_test), dtype=np.float32)

        models = _build_models(seed + fold)
        for model_name, model in models:
            val_pred = _fit_one_model(model_name, model, X_tr, y_tr, X_val, y_val)
            fold_preds += val_pred / len(models)
            fold_test_preds += _predict_one_model(model_name, model, X_te) / len(models)

        oof[val_idx] = fold_preds
        test_preds += fold_test_preds / skf.n_splits
        fold_auc = roc_auc_score(y_val, fold_preds)
        print(f"[CV] seed={seed} fold={fold} auc={fold_auc:.6f}")

    full_auc = roc_auc_score(y, oof)
    print(f"[CV] seed={seed} oof_auc={full_auc:.6f}")
    return oof, test_preds, float(full_auc)


def run_seed_sweep(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, seeds: Sequence[int]) -> Tuple[List[SeedResult], np.ndarray, np.ndarray]:
    seed_results: List[SeedResult] = []
    all_test_preds: List[np.ndarray] = []
    all_oof: List[np.ndarray] = []

    for seed in seeds:
        oof, test_preds, auc = run_seeded_cv(X, y, X_test, seed)
        seed_results.append(
            SeedResult(
                seed=seed,
                oof_auc=auc,
                test_mean=float(np.mean(test_preds)),
                test_std=float(np.std(test_preds)),
            )
        )
        all_oof.append(oof)
        all_test_preds.append(test_preds)

    return seed_results, np.vstack(all_oof), np.vstack(all_test_preds)


def analyze_pseudo_labels(all_test_preds: np.ndarray, lower: float, upper: float, min_agreement: float) -> Dict[str, Any]:
    mean_pred = all_test_preds.mean(axis=0)
    high_conf_mask = (mean_pred <= lower) | (mean_pred >= upper)

    seed_binary = (all_test_preds >= 0.5).astype(int)
    majority = (seed_binary.mean(axis=0) >= 0.5).astype(int)
    unanimity = np.all(seed_binary == majority[None, :], axis=0)
    agreement = (seed_binary == majority[None, :]).mean(axis=0)

    stable_mask = high_conf_mask & (agreement >= min_agreement)
    pseudo_labels = majority[stable_mask]

    return {
        "mean_pred": mean_pred,
        "high_conf_mask": high_conf_mask,
        "majority": majority,
        "unanimity": unanimity,
        "agreement": agreement,
        "stable_mask": stable_mask,
        "pseudo_labels": pseudo_labels,
        "high_confidence_rate": float(high_conf_mask.mean()),
        "unanimity_rate": float(unanimity[high_conf_mask].mean()) if high_conf_mask.any() else 0.0,
        "agreement_rate": float(agreement[high_conf_mask].mean()) if high_conf_mask.any() else 0.0,
        "stable_rate": float(stable_mask.mean()),
        "pseudo_label_rate": float(stable_mask.mean()),
        "pseudo_label_count": int(stable_mask.sum()),
    }


def retrain_with_pseudo_labels(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    pseudo_mask: np.ndarray,
    pseudo_labels: np.ndarray,
    seed: int,
    output_name: str,
) -> str:
    pseudo_rows = X_test.loc[pseudo_mask].copy()
    pseudo_target = pd.Series(pseudo_labels, index=pseudo_rows.index, name=TARGET_COL)

    X_aug = pd.concat([X.reset_index(drop=True), pseudo_rows.reset_index(drop=True)], axis=0, ignore_index=True)
    y_aug = pd.concat([y.reset_index(drop=True), pseudo_target.reset_index(drop=True)], axis=0, ignore_index=True)

    # One more controlled sweep on the augmented set; the goal is a Kaggle submission,
    # not a new OOF estimate.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    test_preds = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_aug, y_aug), start=1):
        X_tr_df = X_aug.iloc[tr_idx].copy()
        X_val_df = X_aug.iloc[val_idx].copy()
        y_tr = y_aug.iloc[tr_idx].to_numpy()
        y_val = y_aug.iloc[val_idx].to_numpy()
        X_tr, X_val, X_te, _, _ = _prepare_fold_matrices(X_tr_df, X_val_df, X_test)

        models = _build_models(seed + 100 + fold)
        fold_test_preds = np.zeros(len(X_test), dtype=np.float32)
        for model_name, model in models:
            _fit_one_model(model_name, model, X_tr, y_tr, X_val, y_val)
            fold_test_preds += _predict_one_model(model_name, model, X_te) / len(models)
        test_preds += fold_test_preds / skf.n_splits

    submission = pd.DataFrame({ID_COL: pd.read_csv(TEST_PATH)[ID_COL], TARGET_COL: test_preds})
    out_path = REPORT_DIR / output_name
    submission.to_csv(out_path, index=False)
    return str(out_path)


def run_probe(
    seeds: Sequence[int],
    lower: float,
    upper: float,
    min_agreement: float,
    min_high_conf_rate: float,
    retrain: bool,
) -> ProbeSummary:
    _, _, y, X, X_test = load_data()
    seed_results, all_oof, all_test_preds = run_seed_sweep(X, y, X_test, seeds)
    avg_oof_auc = float(np.mean([s.oof_auc for s in seed_results]))

    pseudo_stats = analyze_pseudo_labels(all_test_preds, lower=lower, upper=upper, min_agreement=min_agreement)

    retrain_ran = False
    output_submission = None
    output_pseudo_train = None

    if retrain and pseudo_stats["high_confidence_rate"] >= min_high_conf_rate and pseudo_stats["pseudo_label_count"] > 0:
        retrain_ran = True
        output_submission = retrain_with_pseudo_labels(
            X=X,
            y=y,
            X_test=X_test,
            pseudo_mask=pseudo_stats["stable_mask"],
            pseudo_labels=pseudo_stats["pseudo_labels"],
            seed=seeds[0],
            output_name=f"submission_pseudo_label_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        )

        pseudo_rows = X_test.loc[pseudo_stats["stable_mask"]].copy()
        pseudo_rows.insert(0, TARGET_COL, pseudo_stats["pseudo_labels"])
        output_pseudo_train = str(REPORT_DIR / f"pseudo_labeled_rows_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        pseudo_rows.to_csv(output_pseudo_train, index=False)

    summary = ProbeSummary(
        timestamp_utc=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        seeds=list(seeds),
        base_seed_results=seed_results,
        avg_oof_auc=avg_oof_auc,
        high_confidence_rate=pseudo_stats["high_confidence_rate"],
        unanimity_rate=pseudo_stats["unanimity_rate"],
        pseudo_label_rate=pseudo_stats["pseudo_label_rate"],
        pseudo_label_count=pseudo_stats["pseudo_label_count"],
        retrain_ran=retrain_ran,
        output_submission=output_submission,
        output_pseudo_train=output_pseudo_train,
        rag_status={},
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 pseudo-labeling probe for S6E3.")
    parser.add_argument("--research-only", action="store_true", help="Run only the RAG research queries")
    parser.add_argument("--skip-research", action="store_true", help="Skip RAG research and go directly to modeling")
    parser.add_argument("--retrain", action="store_true", help="Retrain on stable pseudo-labeled rows if conditions pass")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 999], help="Seeds for the stability sweep")
    parser.add_argument("--lower", type=float, default=0.15, help="Lower confidence threshold")
    parser.add_argument("--upper", type=float, default=0.85, help="Upper confidence threshold")
    parser.add_argument("--min-agreement", type=float, default=0.80, help="Minimum seed agreement for pseudo labels")
    parser.add_argument("--min-high-conf-rate", type=float, default=0.05, help="Minimum high-confidence fraction required to retrain")
    args = parser.parse_args()

    rag_status: Dict[str, Any] = {}
    if not args.skip_research:
        rag_status = run_rag_queries(RAG_QUERIES)
        if args.research_only:
            print("[DONE] RAG-only mode complete.")
            return

    summary = run_probe(
        seeds=args.seeds,
        lower=args.lower,
        upper=args.upper,
        min_agreement=args.min_agreement,
        min_high_conf_rate=args.min_high_conf_rate,
        retrain=args.retrain,
    )
    summary.rag_status = rag_status

    out_path = REPORT_DIR / "phase1_pseudo_label_probe_summary.json"
    payload = asdict(summary)
    payload["base_seed_results"] = [asdict(x) for x in summary.base_seed_results]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

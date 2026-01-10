"""
Model / Training Pipeline
-------------------------
Consumes artifacts produced by DataPipeline using a run_id.
Trains a model, evaluates it, and saves only model-level artifacts
required by downstream pipelines.

Design principles:
- Artifact-driven (filesystem is the contract)
- No in-memory coupling with previous pipeline
- Single-file, readable research code
"""

import json, uuid
import pickle
import logging
import datetime as dt
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.mda import featImpMDA
from src.evaluation.kFold import cvScore
from src.evaluation.sfi import auxFeatImpSFI, map_featImpSfi
from src.evaluation.feat_pca import compare_imp, orthoFeats


# ============================================================
# Run / Context Initialization
# ============================================================

def init_run(run_id: str):
    """
    Initialize folders and logger for the model pipeline.
    This pipeline writes only model-related artifacts.
    """
    root = Path("runs") / run_id
    artifacts = root / "artifacts"

    MARGS_ID = str(uuid.uuid4().hex[:3])
    margs = root / f"margs_{MARGS_ID}"
    logs = margs / "logs"
    results = artifacts / "results"

    logs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"model-{run_id}")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

    fh = logging.FileHandler(logs / "model_pipeline.log")
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return {
        "root": root,
        "artifacts": artifacts,
        "results": results,
        "margs": margs,
        "logger": logger
    }


# ============================================================
# IO helpers
# ============================================================

def load_dataset(artifacts: Path) -> pd.DataFrame:
    """
    Load dataset produced by DataPipeline.
    """
    path = artifacts / "dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_parquet(path)


def persist_model_metadata(margs: Path, configs, cv_score: float):
    """
    Persist minimal metadata required for lineage and audit.
    """
    meta = {
        "trained_at": dt.datetime.utcnow().isoformat(),
        "model": "RandomForestClassifier",
        "cv_score": cv_score.tolist()
    }

    with open(margs / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    with open(margs / "model_config.json", "w") as f:
        json.dump(vars(configs), f, indent=4, default=str)


# ============================================================
# Core Pipeline
# ============================================================

def model_pipeline(run_id: str, configs) -> None:
    ctx = init_run(run_id)
    logger = ctx["logger"]
    logger.info(f'Initializing the pipeline with path: {ctx}')

    logger.info("-" * 60)
    logger.info(f"MODEL PIPELINE STARTED | RUN_ID={run_id}")

    # --------------------------------------------------------
    # Load dataset (artifact contract with DataPipeline)
    # --------------------------------------------------------
    dataset = load_dataset(ctx["artifacts"])

    X = dataset.drop(["labels", "t1", "weights", "timestamp"], axis=1)
    y = dataset["labels"]
    t1 = dataset["t1"]
    weights = dataset["weights"]

    # --------------------------------------------------------
    # Train / test split (time-ordered)
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    w_train = weights.iloc[:len(X_train)]
    w_test = weights.iloc[len(X_train):]

    # --------------------------------------------------------
    # Model training
    # --------------------------------------------------------
    clf = RandomForestClassifier(**configs.model_params)
    clf.fit(X_train, y_train, sample_weight=w_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds, sample_weight=w_test)
    logger.info(f"Holdout accuracy: {acc:.4f}")

    # --------------------------------------------------------
    # Purged K-Fold Cross Validation
    # --------------------------------------------------------
    cv_score_mean = cvScore(
        clf, X, y, weights,
        t1=t1,
        scoring=configs.scoring,
        pctEmbargo=configs.pctEmbargo,
        cv=configs.cv
    )

    logger.info(f"PurgedKFold CV score: {cv_score_mean}")
    persist_model_metadata(ctx["margs"], configs, cv_score_mean)

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------
    cm = confusion_matrix(
        y_test, preds,
        sample_weight=w_test,
        labels=configs.labels
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="g", cmap="Blues",
        xticklabels=configs.labels,
        yticklabels=configs.labels
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(ctx["results"] / "confusion_matrix.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Feature Importance (MDI)
    # --------------------------------------------------------
    feature_names = X.columns
    mdi = pd.Series(clf.feature_importances_, index=feature_names)

    mdi.sort_values().plot.barh(figsize=(8, 6))
    plt.title("MDI Feature Importance")
    plt.tight_layout()
    plt.savefig(ctx["results"] / "mdi.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Feature Importance (MDA)
    # --------------------------------------------------------
    imp_mda, _ = featImpMDA(
        clf, X, y,
        sample_weight=weights,
        scoring=configs.scoring,
        cv=configs.cv,
        t1=t1,
        pctEmbargo=configs.pctEmbargo
    )

    imp_mda.sort_values("mean").plot.barh(figsize=(10, 6))
    plt.title("MDA Feature Importance")
    plt.tight_layout()
    plt.savefig(ctx["results"] / "mda.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Single Feature Importance (SFI)
    # --------------------------------------------------------
    imp_sfi = auxFeatImpSFI(
        featNames=feature_names,
        clf=clf,
        trnsX=X,
        y=y,
        sample_weight=weights,
        scoring=configs.scoring,
        pctEmbargo=configs.pctEmbargo,
        t1=t1,
        cv=configs.cv
    )

    map_featImpSfi(imp_sfi, folder_path=ctx["results"])

    # --------------------------------------------------------
    # PCA vs MDI comparison (unsupervised vs supervised)
    # --------------------------------------------------------
    _, eVal, _ = orthoFeats(X)
    compare_imp(eVal, mdi=mdi, save_path=ctx["results"])

    # --------------------------------------------------------
    # Persist trained model (next pipeline contract)
    # --------------------------------------------------------
    with open(ctx["artifacts"] / "model.pkl", "wb") as f:
        pickle.dump(clf, f)

    logger.info("MODEL PIPELINE COMPLETED SUCCESSFULLY")



if __name__ == "__main__":
    import argparse
    from config import ModelConfig
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    model_pipeline(run_id=args.run_id, configs=ModelConfig())
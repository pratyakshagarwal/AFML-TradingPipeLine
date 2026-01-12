"""
Backtesting Pipeline (AFML-consistent)
"""

import json, uuid
import logging
import pandas as pd
from pathlib import Path
import datetime as dt

from sklearn.ensemble import RandomForestClassifier

from src.backtest.CPCV import CombinatorialPurgedCV
from src.backtest.engine import AFMLBacktester

from src.data_analysis.data_labeling import TripleBarrierMethod
from src.data_analysis.bootstrapping import SequentialBootstrapping
from src.data_analysis.sample_weights import SampleWeightGenerator
from src.data_analysis.processor import fracDiff_FFD


def init(run_id: str):
    root = Path("runs") / run_id
    BARGS_Id = str(uuid.uuid4().hex[:3])
    artifacts = root / "artifacts"
    bargs = root / f"bargs_{BARGS_Id}"
    logs = bargs / "logs"

    logs.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"testing-{run_id}")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

    fh = logging.FileHandler(logs / "backtesting_pipeline.log")
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return {
        "root": root,
        "artifacts": artifacts,
        "bargs": bargs,
        "logger": logger,
    }


def gSamples(idx, events):
    return [events[i] for i in idx]

def load_dataset(artifacts: Path) -> pd.DataFrame:
    path = artifacts / "unprocessed_data.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def persist_model_metadata(bargs: Path, configs):
    """
    Persist minimal metadata required for lineage and audit.
    """
    meta = {
        "tested_at": dt.datetime.utcnow().isoformat(),
        "technique": "CPCV",
    }

    with open(bargs / "test_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    with open(bargs / "bactest_config.json", "w") as f:
        json.dump(vars(configs), f, indent=4, default=str)


def test(run_id: str, config, dconfigs):
    ctx = init(run_id)
    logger = ctx["logger"]

    logger.info(f"Backtesting Pipeline Started | RUN_ID={run_id}")
    dataset = load_dataset(ctx["artifacts"])

    cpcv = CombinatorialPurgedCV(
        n_splits=config.cpcv_splits,
        n_test_splits=config.cpcv_test_splits,
        pct_embargo=config.cpcv_embargo,
        purge_window=config.purge_window,
    )

    for tr_idx, te_idx in cpcv.split(dataset):
        cpcv_id = "{}_{}".format(te_idx[0], te_idx[-1])
        logger.info(f"Testing Regime Timestamp: {cpcv_id}")

        train_data = dataset.loc[tr_idx].copy().reset_index()
        test_data = dataset.loc[te_idx].copy().reset_index()

        # REMOVE TIME LEAKAGE (PROPERLY)
        train_data = train_data.drop(columns=["timestamp"], errors="ignore")

        # ============================================================
        # 1. TRIPLE BARRIER LABELING (EVENT GENERATION)
        # ============================================================

        labeler = TripleBarrierMethod(
            pt_sl=dconfigs.pt_sl,
            min_ret=dconfigs.min_ret,
            event_specific=dconfigs.event_specific,
        )

        labeler.detect_events(train_data, threshold=dconfigs.threshold)

        trgt = train_data["close"].pct_change().abs()

        n = len(train_data)
        t1 = pd.Series(
            [min(i + dconfigs.h, n - 1) for i in range(n)],
            index=train_data.index,
        )

        labeler.apply_barriers(train_data["close"], trgt=trgt, t1=t1)
        labels = labeler.get_bins(train_data["close"])

        # EVENT INDEX (CANONICAL)
        event_idx = pd.Index(labeler.t_events)

        # ============================================================
        # 2. SAMPLE WEIGHTS (EVENT-BASED)
        # ============================================================

        t1_series = pd.Series(labeler.events_["t1"], index=labeler.t_events)

        w_gen = SampleWeightGenerator(
            closeIdx=event_idx,
            t1=t1_series,
            molecule=event_idx,
        )

        weights_raw = w_gen._mpSampleW(train_data["close"])
        clfW = w_gen.getTimeDecay(weights_raw, clfLastW=dconfigs.clfLastW)

        # ============================================================
        # 3. FRACTIONAL DIFFERENCING (BAR-LEVEL)
        # ============================================================

        df_ffd = fracDiff_FFD(
            train_data[dconfigs.fracdiff_cols],
            d=dconfigs.d,
            thres=dconfigs.thres,
        )

        for col in dconfigs.fracdiff_cols:
            train_data.loc[df_ffd.index, col] = df_ffd[col]

        # ============================================================
        # 4. REMOVE EVENTS BROKEN BY FRACDIFF
        # ===========================================================

        missing = train_data.index.difference(df_ffd.index)
        valid_idx = labeler.events_.index.drop(missing, errors="ignore") 
        t1_valid = t1.loc[valid_idx]
        # ============================================================
        # 5. SEQUENTIAL BOOTSTRAPPING (EVENT-LEVEL)
        # ============================================================

        bootstrapper = SequentialBootstrapping(
            barIx=valid_idx,
            t1=t1_valid,
        )

        samples = bootstrapper._seqbootstrap_numba_tqdm(
            sLength=dconfigs.sLength
        )

        samples = gSamples(idx=samples, events=labeler.t_events)
        Xtrain = train_data.loc[samples]
        ytrain = labels['bin'].loc[samples]
        weights = clfW.loc[samples]

        # ============================================================
        # 6. MODEL FIT
        # ============================================================

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            n_jobs=-1,
            random_state=42,
        )

        clf.fit(Xtrain, ytrain, sample_weight=weights)

        # ============================================================
        # 7. BACKTEST (OUT-OF-SAMPLE)
        # ============================================================

        bt = AFMLBacktester(test_data, config, model=clf)
        summary, result_df = bt.run()
        logger.info(f"Summary: {summary}")
        filename = f"{cpcv_id}.csv"
        pth = ctx['bargs'] / filename
        pth.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(pth)
        
    persist_model_metadata(ctx["bargs"], configs=config)
    logger.info(f"saving pipeline metadata and configs at path: {ctx['bargs']}")
    

if __name__ == "__main__":
    import argparse
    from config import BTConfig, DataConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    test(
        args.run_id,
        config=BTConfig(),
        dconfigs=DataConfig(),
    )
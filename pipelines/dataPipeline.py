import os
import json
import uuid
import asyncio
import logging
import pandas as pd
import datetime as dt
from typing import Any

from src.data_analysis.featuresEng import make_features
from src.data_analysis.fetcher import DataFetcher
from src.data_analysis.processor import BarGenerator, fracDiff_FFD
from src.data_analysis.data_labeling import TripleBarrierMethod
from src.data_analysis.sample_weights import SampleWeightGenerator
from src.data_analysis.bootstrapping import SequentialBootstrapping


# -------------------- RUN SETUP --------------------

RUN_ID = str(uuid.uuid4().hex[:8])
RUN_ROOT = os.path.join("runs", RUN_ID)
DARGS_DIR = os.path.join(RUN_ROOT, "dargs")
LOG_DIR = os.path.join(DARGS_DIR, "logs")
DATASET_DIR = os.path.join(RUN_ROOT, "artifacts")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)


# -------------------- LOGGING --------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# -------------------- METADATA --------------------

def persist_run_metadata(configs: Any) -> None:
    meta = {
        "run_id": RUN_ID,
        "created_at": dt.datetime.utcnow().isoformat(),
        "pipeline": "data_pipeline",
        "config_class": configs.__class__.__name__
    }

    with open(os.path.join(DARGS_DIR, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    with open(os.path.join(DARGS_DIR, "configs.json"), "w") as f:
        json.dump(vars(configs), f, indent=4, default=str)


# -------------------- PIPELINE --------------------

async def fetch_data(configs: Any) -> None:
    fetcher = DataFetcher(
        asset=configs.asset,
        type=configs.type,
        interval_ms=configs.interval_ms
    )
    await fetcher.fetch(nRows=configs.nRows)
    logger.info(f"Fetched {configs.nRows:,} rows for {configs.asset}")


def gSamples(idx, events, removal):
    return [events[i] for i in idx if i not in removal]


def data_pipeline(configs: Any) -> None:
    logger.info("-" * 60)
    logger.info(f"DATA PIPELINE STARTED | RUN_ID={RUN_ID}")

    persist_run_metadata(configs)

    # Step 1: Fetch data
    asyncio.run(fetch_data(configs))

    raw_dir = f"data/{configs.asset}"
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Missing raw data directory: {raw_dir}")

    # Step 2: Generate bars
    bar_files = []
    for name in os.listdir(raw_dir):
        if not name.endswith(".jsonl"):
            continue

        path = os.path.join(raw_dir, name)
        df = pd.read_json(path, lines=True).drop_duplicates(subset=["trade_id"])

        bg = BarGenerator(threshold=configs.threshold, bar_type=configs.bar_type)
        df_bars = bg.generate(df)

        date = name.split("_")[0]
        bar_path = os.path.join(
            DATASET_DIR, f"{date}_{configs.bar_type}bars_{configs.threshold}.csv"
        )
        df_bars.to_csv(bar_path, index=False)
        bar_files.append(bar_path)

        logger.info(f"Bars saved -> {bar_path}")

    # Step 3: Combine + features
    dfs = [pd.read_csv(f).dropna() for f in bar_files]
    df = (
        pd.concat(dfs, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df = make_features(df, window=configs.window)
    logger.info(f"Feature matrix shape: {df.shape}")

    pth = os.path.join(DATASET_DIR, "unprocessed_data.parquet")
    df.to_parquet(path=pth)

    # Step 4: Triple Barrier Labeling
    labeler = TripleBarrierMethod(
        pt_sl=configs.pt_sl,
        min_ret=configs.min_ret,
        event_specific=configs.event_specific
    )

    labeler.detect_events(df, threshold=configs.threshold)
    trgt = df["close"].pct_change().abs()

    n = len(df)
    t1 = pd.Series([min(i + configs.h, n - 1) for i in range(n)], name="t1")

    labeler.apply_barriers(df["close"], trgt=trgt, t1=t1)
    labels = labeler.get_bins(df["close"])

    # Step 5: Sample weights
    closeIdx = pd.Index(labeler.t_events)
    t1_series = pd.Series(labeler.events_["t1"], index=labeler.t_events)

    w_gen = SampleWeightGenerator(
        closeIdx=closeIdx,
        t1=t1_series,
        molecule=labeler.t_events
    )

    weights_raw = w_gen._mpSampleW(df["close"])
    clfW = w_gen.getTimeDecay(weights_raw, clfLastW=configs.clfLastW)

    # Step 6: Bootstrapping
    bootstrapper = SequentialBootstrapping(barIx=closeIdx, t1=t1_series)
    samples = bootstrapper._seqbootstrap_numba_tqdm(sLength=configs.sLength)

    # Step 7: Fractional differencing
    df_ffd = fracDiff_FFD(
        df[configs.fracdiff_cols],
        d=configs.d,
        thres=configs.thres
    )

    for col in configs.fracdiff_cols:
        df[col] = df_ffd[col]

    missing = df.index.difference(df_ffd.index)
    valid_idx = gSamples(samples, labeler.t_events, missing)

    # Step 8: Final dataset
    dataset = (
        df.loc[valid_idx]
        .sort_index()
        .drop_duplicates()
        .assign(
            labels=labels["bin"].loc[valid_idx],
            weights=clfW.loc[valid_idx],
            t1=labeler.events_["t1"].loc[valid_idx]
        )
    )

    dataset_path = os.path.join(DATASET_DIR, "dataset.parquet")
    dataset.to_parquet(dataset_path)

    logger.info(f"FINAL DATASET SAVED -> {dataset_path}")
    logger.info(f"PIPELINE COMPLETED | RUN_ID={RUN_ID}")


# -------------------- ENTRYPOINT --------------------

if __name__ == "__main__":
    from config import DataConfig
    data_pipeline(DataConfig())

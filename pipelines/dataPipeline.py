import os
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


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{dt.datetime.now().strftime("%Y%m%d%H%M")}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def fetch_data(configs: Any) -> None:
    """Fetch raw trade/order data asynchronously."""
    try:
        fetcher = DataFetcher(asset=configs.asset, type=configs.type, interval_ms=configs.interval_ms)
        await fetcher.fetch(nRows=configs.nRows)
        logger.info(f"Fetched {configs.nRows:,} rows for {configs.asset}")
    except Exception as e:
        logger.exception(f"Error fetching data: {e}")
        raise

def gSamples(idx, events, removal):
    """Filter out indices corresponding to removed samples."""
    return [events[i] for i in idx if i not in removal]

# returns a dataset which contains features, labels and weights
def data_pipeline(configs: Any) -> pd.DataFrame:
    logger.info(f"{'-' * 50}")
    logger.info(f"Data Pipeline started at {dt.datetime.now().strftime("%Y%m%d%H%M")}")
    logger.info(f"Asset: {configs.asset} | Bar type: {configs.bar_type}")

    # Step 1: Data fetching
    asyncio.run(fetch_data(configs))

    raw_dir = f"data/{configs.asset}"
    if not os.path.exists(raw_dir):
        logger.error(f"Data directory not found: {raw_dir}")
        raise FileNotFoundError(f"Missing directory {raw_dir}")

    # Step 2: Generate bars
    bar_files = []
    try:
        for name in os.listdir(raw_dir):
            if not name.endswith(".jsonl"):
                continue

            path = os.path.join(raw_dir, name)
            logger.info(f"Processing raw file: {path}")
            df = pd.read_json(path, lines=True).drop_duplicates(subset=["trade_id"])
            logger.info(f"Raw shape: {df.shape}")

            # Create bars
            bg = BarGenerator(threshold=configs.threshold, bar_type=configs.bar_type)
            df_bars = bg.generate(df)
            logger.info(f"Bars created: {df_bars.shape}")

            date = name.split("_")[0]
            bar_path = os.path.join(raw_dir, f"{date}_{configs.bar_type}bars{configs.threshold}.csv")
            df_bars.to_csv(bar_path, index=False)
            bar_files.append(bar_path)
            logger.info(f"Saved bars -> {bar_path}")

        logger.info(f"Total {len(bar_files)} bar files generated")

    except Exception as e:
        logger.exception(f"Error generating bars: {e}")
        raise

    # Step 3: Combine bar files and feature engineering
    try:
        csv_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".csv")]
        logger.info(f"Combining {len(csv_files)} CSV files")

        dfs = []
        for f in csv_files:
            temp = pd.read_csv(f).dropna()
            dfs.append(temp)
            logger.info(f"Loaded {f}, shape={temp.shape}")

        df = pd.concat(dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Combined dataset shape: {df.shape}")

        # Feature creation
        df = make_features(df, window=configs.window)
        logger.info(f"Feature matrix created with shape: {df.shape}")

    except Exception as e:
        logger.exception(f"Error during feature generation: {e}")
        raise

    # Step 4: Triple Barrier Labeling
    try:
        labeler = TripleBarrierMethod(
            pt_sl=configs.pt_sl,
            min_ret=configs.min_ret,
            event_specific=configs.event_specific
        )
        # detect special events on which we will place trades
        labeler.detect_events(df, threshold=configs.threshold)
        logger.info(f"Detected {len(labeler.t_events)} events")


        trgt = df["close"].pct_change().abs()
        n = len(df)
        # generate vertical bars
        t1 = pd.Series([min(i + configs.h, n - 1) for i in range(n)], name="t1")

        # apply vertical (termination) and horizontal barriers (stop loss and profit cap)
        labeler.apply_barriers(df["close"], trgt=trgt, t1=t1)
        labels = labeler.get_bins(df["close"])
        logger.info(f"Labeled {labels['bin'].notna().sum()} valid bins")

    except Exception as e:
        logger.exception(f"Error during labeling: {e}")
        raise

    # Step 5: Compute sample weights
    try:
        closeIdx = df.index
        t1_series = pd.Series(labeler.events_["t1"], index=labeler.t_events, dtype="float64")

        # compute sample weights bases on uniquessness and timedecaying 
        w_gen = SampleWeightGenerator(closeIdx=closeIdx, t1=t1_series, molecule=labeler.t_events)
        weights_raw = w_gen._mpSampleW(df["close"])
        clfW = w_gen.getTimeDecay(weights_raw, clfLastW=configs.clfLastW)
        logger.info(f"Computed sample weights for {len(clfW)} events")

    except Exception as e:
        logger.exception(f"Error generating sample weights: {e}")
        raise

    # Step 6: Bootstrapping for sample selection
    try:
        # bootstapping using numba
        bootstrapper = SequentialBootstrapping(barIx=closeIdx, t1=t1_series)
        samples = bootstrapper._seqbootstrap_numba_tqdm(sLength=configs.sLength)
        logger.info(f"Bootstrapped {len(samples)} samples")

    except Exception as e:
        logger.exception(f"Error in bootstrapping: {e}")
        raise

    # Step 7: Fractional differencing
    try:
        logger.info(f"Applying fractional differencing (d={configs.d}, thres={configs.thres})")
        # fractional differentialtion the features which contains randomness and noise
        df_ffd = fracDiff_FFD(df[configs.fracdiff_cols], d=configs.d, thres=configs.thres)
        for col in configs.fracdiff_cols:
            df[col] = df_ffd[col]

        missing_indexes = df.index.difference(df_ffd.index)
        valid_indexes = gSamples(samples, labeler.t_events, missing_indexes)
        logger.info(f"FFD complete -> dropped {len(missing_indexes)} rows, kept {len(valid_indexes)}")

    except Exception as e:
        logger.exception(f"Error during fractional differencing: {e}")
        raise

    # Step 8: Final dataset assembly
    try:
        dataset = (
            df.loc[valid_indexes]
            .sort_index()
            .drop_duplicates()
            .assign(
                labels=labels["bin"].loc[valid_indexes].dropna(),
                weights=clfW.loc[valid_indexes], 
                t1=labeler.events_['t1'].loc[valid_indexes]
            )
        )
        logger.info(f"Final dataset ready -> shape={dataset.shape}")
        logger.info(f"Pipeline completed successfully for {configs.asset}")

        return dataset, logger

    except Exception as e:
        logger.exception(f"Error during dataset assembly: {e}")
        raise


if __name__ == "__main__":
    from config import DataConfig
    dataset = data_pipeline(configs=DataConfig())
    print(dataset.head())
    print(f"Dataset shape: {dataset.shape}")

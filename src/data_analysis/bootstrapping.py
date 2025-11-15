import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit, njit, prange
import matplotlib.pyplot as plt
from src.data_analysis.data_labeling import TripleBarrierMethod

from numba import njit, prange
import numpy as np
from tqdm import tqdm

@njit(parallel=True)
def seqBootstrap_step_numba(indMat, concurrency, mask):
    n_obs, n_events = indMat.shape
    avgU = np.zeros(n_events, np.float32)
    total = np.float32(0.0)

    for i in prange(n_events):
        if mask[i] == 0:
            val = np.float32(0.0)
            count = 0
            for j in range(n_obs):
                if indMat[j, i] > 0:
                    val += indMat[j, i] / max(concurrency[j], 1)
                    count += 1
            if count > 0:
                avgU[i] = val / count
                total += avgU[i]

    if total == 0:
        prob = np.ones(n_events, np.float32) / n_events
    else:
        prob = avgU / total

    r = np.random.random()
    cum = 0.0
    for i in range(prob.shape[0]):
        cum += prob[i]
        if r < cum:
            return i
    return prob.shape[0] - 1


class SequentialBootstrapping:
    def __init__(self, barIx, t1):
        self.indMat = self.getIndMatrix(barIx, t1)

    def _seqbootstrap_numba_tqdm(self, sLength: int = None):
        if isinstance(self.indMat, pd.DataFrame):
            indMat = self.indMat.values.astype(np.float32)
        else:
            indMat = self.indMat.astype(np.float32)

        n_obs, n_events = indMat.shape
        if sLength is None:
            sLength = n_events

        concurrency = indMat.sum(axis=1).astype(np.float32)
        mask = np.zeros(n_events, np.int32)
        phi = np.zeros(sLength, np.int32)

        for step in tqdm(range(sLength), desc="Sequential Bootstrapping"):
            chosen = seqBootstrap_step_numba(indMat, concurrency, mask)
            phi[step] = chosen
            mask[chosen] = 1

        return phi

    @njit
    def weighted_random_choice(self, prob):
        """
    Equivalent to np.random.choice with p=prob (Numba compatible).
    """
        r = np.random.random()
        cum = 0.0
        for i in range(prob.shape[0]):
            cum += prob[i]
            if r < cum: return i
        return prob.shape[0] - 1
    
    def _getAvgUniq(self):
        # Average Uniqueness from Indication Matrix
        c = self.indMat.sum(axis=1) # concurrency
        u = self.indMat.div(c, axis=0) # uniqueness
        avgU = u[u>0].mean() # average uniqueness
        return avgU

    def getIndMatrix(self, barIx, t1):
        # get indication matrix
        indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
        for i, tIn in enumerate(t1.index):
            tOut = t1.loc[tIn]
            indM.loc[tIn:tOut, i] = 1
        return indM
    

if __name__ == '__main__':
    df = pd.read_csv('data/btcusdt/2025102617_tradebars.csv').dropna()
    labeler = TripleBarrierMethod(min_ret=0.0)
    labeler.detect_events(df, threshold=1)

    plt.figure(figsize=(16,8))
    plt.plot(df['close'])
    plt.plot(df['close'][labeler.t_events], 'ro', markersize=2.5)
    plt.show()

    trgt = df['close'].pct_change()
    h, n = 5, len(df)
    t1 = pd.Series([i + h if i + h < n else n-1 for i in range(n)], name='t1')
    labeler.apply_barriers(df['close'], trgt=trgt, t1=t1)
    
    labels = labeler.get_bins(df['close'])
    print(labels.head())

    closeId, t1 = df.index, pd.Series(labeler.events_['t1'], index=labeler.t_events, dtype='float64')
    bootstrapper = SequentialBootstrapping(barIx=closeId, t1=t1)
    print(bootstrapper.indMat)

    samples = bootstrapper._seqbootstrap_numba_tqdm(sLength=5)
    print(samples)
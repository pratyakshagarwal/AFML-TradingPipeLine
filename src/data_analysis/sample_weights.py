import numpy as np
import pandas as pd
from src.data_analysis.data_labeling import TripleBarrierMethod

class SampleWeightGenerator:
    def __init__(self, closeIdx, t1, molecule):
        self.t1, self.molecule = t1, molecule
        self.count = self.mpNumCoEvents(closeIdx, t1, molecule)
    
    def mpNumCoEvents(self, closeIdx, t1, molecule):
        """
        Compute the number of concurrent events per bar.
        + molecule[0] is the date of the first event on which the weight will be computed
        + molecule[-1] is the date of the last event on which the weight will be computed 
        Any event that starts before t1[molecule].max() impacts the count.
        """
        # 1) find events that span the period [molecule[0], molecule[-1]]
        t1 = t1.fillna(closeIdx[-1])                 # unclosed events still must impact other weights
        t1 = t1[t1 >= molecule[0]]                   # events that end at or after molecule[0]
        t1 = t1.loc[:t1[molecule].max()]             # events that start at or before t1[molecule].max()

        # 2) count events spanning a bar
        iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
        count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1] + 1])
        for tIn in t1.index.tolist():
            tOut = t1.loc[tIn]
            count.loc[tIn:tOut] += 1.0

        return count.loc[molecule[0]:t1[molecule].max()]
    
    
    def _mpSampleW(self, close):
        # Derive sample weights by return attribution
        ret=np.log(close).diff()
        wght=pd.Series(index=self.molecule)
        for tIn in self.molecule:
            tOut=self.t1.loc[tIn]
            wght.loc[tIn]=(ret.loc[tIn:tOut]/self.count.loc[tIn:tOut]).sum()
        return wght.abs()
    
    def getTimeDecay(self, tW, clfLastW=1):
        # apply piecewise linear time decay to observerd uniqueness (tW)
        # newest observations get weight = 1, oldest pbservation get weight = clfLastW
        clfW = tW.sort_index().cumsum()
        if clfLastW>=0: slope=(1.-clfLastW)/clfW.iloc[-1]
        else: slope=1./ ((clfLastW+1)*clfW.iloc[-1])
        const=1.-slope*clfW.iloc[-1]
        clfW=const + slope*clfW
        clfW[clfW<0]=0
        print(const, slope)
        return clfW
    
if __name__ == '__main__':
    df = pd.read_csv('data/btcusdt/2025102617_tradebars.csv').dropna()
    labeler = TripleBarrierMethod(min_ret=0.0)
    labeler.detect_events(df, threshold=1)

    trgt = df['close'].pct_change()
    h, n = 5, len(df)
    t1 = pd.Series([i + h if i + h < n else n-1 for i in range(n)], name='t1')
    labeler.apply_barriers(df['close'], trgt=trgt, t1=t1)
    
    labels = labeler.get_bins(df['close'])
    print(labels.head())

    closeId, t1 = df.index, pd.Series(labeler.events_['t1'], index=labeler.t_events, dtype='float64')
    
    wghts_generator = SampleWeightGenerator(closeIdx=closeId, t1=t1, molecule=labeler.t_events)
    print(wghts_generator.count)
    wghts = wghts_generator._mpSampleW(df['close'])
    print(wghts)
    clfW = wghts_generator.getTimeDecay(wghts, clfLastW=0.8)
    print(clfW)

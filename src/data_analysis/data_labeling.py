import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TripleBarrierMethod:
    def __init__(self, pt_sl=(1, 1), min_ret=0.01, event_specific=True):
        self.pt_sl = pt_sl
        self.min_ret = min_ret
        self.event_specific = event_specific
        self.events_, self.barriers_, self.bins_ = None, None, None

    def detect_events(self, data, column='close', threshold=1.0):
        """Detect timestamps where cumulative move exceeds threshold."""
        self.t_events, s_pos, s_neg = [], 0.0, 0.0
        diff = data[column].diff()
        for i in diff.index[1:]:
            s_pos, s_neg = max(0, s_pos + diff.loc[i]), min(0, s_neg + diff.loc[i])
            if s_pos > threshold or s_neg < -threshold:
                s_pos = s_neg = 0.0
                self.t_events.append(i)

    def apply_barriers(self, prices, trgt, side=None, t1=None):
        """Core triple barrier application."""
        pt_sl = self.pt_sl
        if t1 is None: t1 = pd.Series(pd.NaT, index=self.t_events)

        if side is None: side_, pt_sl_= pd.Series(1., index=self.t_events), [pt_sl[0], pt_sl[0]] 
        else:  side_, pt_sl_ = side.loc[self.t_events], pt_sl[:2]

        trgt = trgt.loc[self.t_events]
        trgt = trgt[trgt > self.min_ret]
        events_df = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])

        barriers = self._compute_barriers(prices, events_df, pt_sl_, events_df.index)
        events_df['t1'] = barriers.dropna(how='all').min(axis=1)

        if side is None: events_df = events_df.drop('side', axis=1)

        self.events_ = events_df
        self.barriers_ = barriers

    def _compute_barriers(self, prices, events, pt_sl, molecule):
        """Internal helper — the barrier search loop."""
        raw = prices
        out = events[['t1']].copy(deep=True)

        pt = pt_sl[0] * events['trgt'] if pt_sl[0] > 0 else pd.Series(index=events.index)
        sl = -pt_sl[1] * events['trgt'] if pt_sl[1] > 0 else pd.Series(index=events.index)

        for loc, t1 in zip(events.index, events['t1'].fillna(raw.index[-1])):
            df0 = raw.loc[loc:t1]
            df0 = (df0 / raw.loc[loc] - 1) * events.at[loc, 'side']
            out.at[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
            out.at[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
        return out

    def get_bins(self, prices):
        """Generate return bins for previously computed events."""
        if self.events_ is None:
            raise ValueError("No events found. Run `apply_barriers()` first.")
        events = self.events_.dropna(subset=['t1'])
        px = prices.reindex(events.index.union(events['t1'].values).drop_duplicates(), method='bfill')
        out = pd.DataFrame(index=events.index)
        out['ret'] = px.loc[events['t1'].values].values / px.loc[events.index] - 1
        if 'side' in events: out['ret'] *= events['side']
        out['bin'] = np.sign(out['ret'])
        if 'side' in events: out.loc[out['ret'] <= 0, 'bin'] = 0
        self.bins_ = out
        return out
    
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

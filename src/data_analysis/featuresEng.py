import numpy as np
def make_features(df, window):
    df = df.drop(['open', 'high', 'low'], axis=1)
    # log return 
    df['ret'] = np.log(df['close'] / df['close'].shift(1))

    df['rw_mean'] = df['ret'].rolling(window).mean().shift(1)
    df['rw_std'] = df['ret'].rolling(window).std().shift(1)
    df['rw_std'] = df['rw_std'].replace(0, 1e-8)

    # Z-score (using only past info)
    df['zscore'] = (df['ret'] - df['rw_mean']) / df['rw_std'] + 1e-8

    # EMAs (momentum)
    df['ema_fast'] = df['close'].ewm(span=window, adjust=False).mean().shift(1)
    df['ema_slow'] = df['close'].ewm(span=window*3, adjust=False).mean().shift(1)
    df['mom'] = df['ema_fast'] - df['ema_slow']

    # directional sign feature
    df['ret_sign'] = np.sign(df['ret']).shift(1)
    df['ret_sign_sum'] = df['ret_sign'].rolling(window).sum().shift(1)

    # lagged returns
    for lag in range(1, 4):
        df[f'ret_lag{lag}'] = df['ret'].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas_ta as ta

FEATURE_COLS = [
    "return_1d", "log_return", "hv_5", "hv_20",
    "rsi", "macd", "macd_hist",
    "bb_pct", "bb_width", "atr",
    "ema12_dist", "ema26_dist",
    "vol_ratio", "obv_change",
]


def build_features(df):
    d = df.copy()
    c = d["Close"]
    h = d["High"]
    l = d["Low"]
    v = d["Volume"]

    d["return_1d"]  = c.pct_change()
    d["log_return"] = np.log(c / c.shift(1))
    d["hv_5"]       = d["log_return"].rolling(5).std()
    d["hv_20"]      = d["log_return"].rolling(20).std()

    d["rsi"] = ta.rsi(c, length=14)

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    d["macd"]      = macd["MACD_12_26_9"]
    d["macd_hist"] = macd["MACDh_12_26_9"]

    bb = ta.bbands(c, length=20, std=2)
    d["bb_pct"]   = (c - bb["BBL_20_2.0"]) / (bb["BBU_20_2.0"] - bb["BBL_20_2.0"])
    d["bb_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / bb["BBM_20_2.0"]

    d["atr"] = ta.atr(h, l, c, length=14) / c

    d["ema12_dist"] = (ta.ema(c, 12) - c) / c
    d["ema26_dist"] = (ta.ema(c, 26) - c) / c

    d["vol_ratio"]  = v / v.rolling(20).mean()
    d["obv_change"] = ta.obv(c, v).pct_change()

    return d.dropna()


def build_targets(df, lookahead=1):
    future_return = df["Close"].shift(-lookahead) / df["Close"] - 1
    direction = (future_return > 0).astype(int).rename("direction")
    magnitude = (future_return * 100).rename("magnitude")
    return direction, magnitude


def make_sequences(features, direction, magnitude, seq_len=60):
    X, y_dir, y_mag = [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        y_dir.append(direction[i])
        y_mag.append(magnitude[i])
    return (
        np.array(X, dtype=np.float32),
        np.array(y_dir, dtype=np.float32),
        np.array(y_mag, dtype=np.float32),
    )

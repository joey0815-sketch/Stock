"""
features.py — 技術指標特徵工程，產生模型輸入特徵與預測目標
"""
import numpy as np
import pandas as pd
import pandas_ta as ta


# ─────────────────────────────────────────────
#  特徵計算
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    從 OHLCV DataFrame 計算技術指標特徵。

    特徵清單（共 15 個）：
      日報酬率、對數報酬率、5/20 日波動率、
      RSI(14)、MACD、MACD柱狀、
      Bollinger %B、BB寬度、ATR(14)、
      EMA12/26、成交量比率、OBV變化率
    """
    d = df.copy()
    c = d["Close"]
    h = d["High"]
    l = d["Low"]
    v = d["Volume"]

    # 報酬率
    d["return_1d"]   = c.pct_change()
    d["log_return"]  = np.log(c / c.shift(1))

    # 歷史波動率
    d["hv_5"]  = d["log_return"].rolling(5).std()
    d["hv_20"] = d["log_return"].rolling(20).std()

    # RSI
    d["rsi"] = ta.rsi(c, length=14)

    # MACD
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    d["macd"]      = macd["MACD_12_26_9"]
    d["macd_hist"] = macd["MACDh_12_26_9"]

    # Bollinger Bands
    bb = ta.bbands(c, length=20, std=2)
    d["bb_pct"]   = (c - bb["BBL_20_2.0"]) / (bb["BBU_20_2.0"] - bb["BBL_20_2.0"])
    d["bb_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / bb["BBM_20_2.0"]

    # ATR
    d["atr"] = ta.atr(h, l, c, length=14) / c

    # EMA 差距（相對收盤價）
    d["ema12_dist"] = (ta.ema(c, 12) - c) / c
    d["ema26_dist"] = (ta.ema(c, 26) - c) / c

    # 成交量比率
    d["vol_ratio"] = v / v.rolling(20).mean()

    # OBV 變化率
    obv = ta.obv(c, v)
    d["obv_change"] = obv.pct_change()

    return d.dropna()


FEATURE_COLS = [
    "return_1d", "log_return", "hv_5", "hv_20",
    "rsi", "macd", "macd_hist",
    "bb_pct", "bb_width", "atr",
    "ema12_dist", "ema26_dist",
    "vol_ratio", "obv_change",
]


# ─────────────────────────────────────────────
#  目標變數：方向 + 幅度
# ─────────────────────────────────────────────

def build_targets(df: pd.DataFrame, lookahead: int = 1):
    """
    計算預測目標。

    Args:
        df:        含 Close 欄位的 DataFrame
        lookahead: 預測幾天後（預設 1 = 明日）

    Returns:
        direction: 1(漲) / 0(跌) 的 Series
        magnitude: 漲跌幅度（含正負號）的 Series，單位 %
    """
    future_return = df["Close"].shift(-lookahead) / df["Close"] - 1
    magnitude  = (future_return * 100).rename("magnitude")
    direction  = (future_return > 0).astype(int).rename("direction")
    return direction, magnitude


# ─────────────────────────────────────────────
#  序列切割（for LSTM）
# ─────────────────────────────────────────────

def make_sequences(features: np.ndarray, direction: np.ndarray,
                   magnitude: np.ndarray, seq_len: int = 60):
    """
    將特徵陣列切成 (樣本數, seq_len, n_features) 的 3D 張量。
    """
    X, y_dir, y_mag = [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        y_dir.append(direction[i])
        y_mag.append(magnitude[i])
    return np.array(X, dtype=np.float32), \
           np.array(y_dir, dtype=np.float32), \
           np.array(y_mag, dtype=np.float32)

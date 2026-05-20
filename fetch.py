"""
fetch.py — 下載股價歷史資料
"""
import yfinance as yf
import pandas as pd


def fetch_ohlcv(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    """
    下載 OHLCV 日線資料。

    Args:
        ticker: 股票代號，例如 'AAPL'、'2330.TW'
        start:  開始日期 'YYYY-MM-DD'
        end:    結束日期（預設今天）

    Returns:
        DataFrame，columns: Open, High, Low, Close, Volume
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"查無資料：{ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df

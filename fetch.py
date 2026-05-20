# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd


def fetch_ohlcv(ticker, start, end=None):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df

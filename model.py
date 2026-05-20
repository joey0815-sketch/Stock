"""
model.py — 雙輸出 LSTM：同時預測漲跌方向（分類）與幅度（回歸）
"""
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from features import (
    build_features, build_targets, make_sequences, FEATURE_COLS
)
from fetch import fetch_ohlcv


SEQ_LEN   = 60   # 回溯天數
SAVE_DIR  = "saved_model"


# ─────────────────────────────────────────────
#  建立雙輸出模型
# ─────────────────────────────────────────────

def build_model(n_features: int) -> Model:
    """
    共用 LSTM 主幹 → 分叉為兩個輸出 head：
      - direction_out : sigmoid，漲跌二元分類
      - magnitude_out : linear，漲跌幅度回歸
    """
    inp = Input(shape=(SEQ_LEN, n_features), name="input")

    # 共用主幹
    x = LSTM(128, return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = LSTM(64, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation="relu")(x)
    x = Dropout(0.1)(x)

    # 分類 head（漲 or 跌）
    direction_out = Dense(1, activation="sigmoid", name="direction")(x)

    # 回歸 head（漲跌幅度 %）
    magnitude_out = Dense(1, activation="linear", name="magnitude")(x)

    model = Model(inputs=inp, outputs=[direction_out, magnitude_out])
    model.compile(
        optimizer=Adam(1e-3),
        loss={
            "direction":  "binary_crossentropy",
            "magnitude":  "huber",
        },
        loss_weights={
            "direction":  1.0,
            "magnitude":  0.5,
        },
        metrics={
            "direction":  "accuracy",
            "magnitude":  "mae",
        },
    )
    return model


# ─────────────────────────────────────────────
#  訓練
# ─────────────────────────────────────────────

def train(ticker: str, start: str, end: str = None):
    """
    下載資料 → 特徵工程 → 訓練模型 → 儲存。
    """
    print(f"\n⬇  下載 {ticker} 資料中…")
    df = fetch_ohlcv(ticker, start, end)

    print("🔧 計算特徵…")
    df = build_features(df)
    direction, magnitude = build_targets(df, lookahead=1)

    # 去掉最後一列（lookahead 沒有目標值）
    df = df.iloc[:-1]
    direction = direction.iloc[:-1].values
    magnitude = magnitude.iloc[:-1].values

    # 標準化特徵
    scaler = StandardScaler()
    feat = scaler.fit_transform(df[FEATURE_COLS].values)

    # 切序列
    X, y_dir, y_mag = make_sequences(feat, direction, magnitude, SEQ_LEN)
    print(f"   樣本數：{len(X)}  特徵數：{X.shape[2]}")

    # Train / Val split
    X_tr, X_val, yd_tr, yd_val, ym_tr, ym_val = train_test_split(
        X, y_dir, y_mag, test_size=0.15, shuffle=False
    )

    model = build_model(X.shape[2])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
    ]

    print("🧠 訓練中…")
    model.fit(
        X_tr,
        {"direction": yd_tr, "magnitude": ym_tr},
        validation_data=(X_val, {"direction": yd_val, "magnitude": ym_val}),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # 儲存模型與 scaler
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(os.path.join(SAVE_DIR, f"{ticker}_model.keras"))
    with open(os.path.join(SAVE_DIR, f"{ticker}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✅ 模型已儲存至 {SAVE_DIR}/")
    return model, scaler


# ─────────────────────────────────────────────
#  預測
# ─────────────────────────────────────────────

def predict(ticker: str) -> dict:
    """
    載入已訓練模型，預測明日漲跌方向與幅度。

    Returns:
        {
            "direction":   "漲" | "跌",
            "magnitude":   float（如 +2.37 或 -1.05）,
            "confidence":  float（0~1）,
            "current_price": float,
            "predicted_price": float,
        }
    """
    model_path  = os.path.join(SAVE_DIR, f"{ticker}_model.keras")
    scaler_path = os.path.join(SAVE_DIR, f"{ticker}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型，請先執行：python predict.py train --ticker {ticker}")

    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # 取最近 (SEQ_LEN + 200) 天，確保指標計算足夠
    from datetime import datetime, timedelta
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=SEQ_LEN + 300)).strftime("%Y-%m-%d")

    df = fetch_ohlcv(ticker, start, end)
    df = build_features(df)

    feat   = scaler.transform(df[FEATURE_COLS].values)
    seq    = feat[-SEQ_LEN:][np.newaxis, :, :].astype(np.float32)   # (1, 60, F)

    dir_prob, mag_pred = model.predict(seq, verbose=0)
    dir_prob  = float(dir_prob[0][0])   # 漲的機率
    magnitude = float(mag_pred[0][0])   # 預測幅度（%）

    current_price   = float(df["Close"].iloc[-1])
    predicted_price = current_price * (1 + magnitude / 100)

    direction  = "漲" if dir_prob >= 0.5 else "跌"
    confidence = dir_prob if dir_prob >= 0.5 else 1 - dir_prob

    return {
        "direction":       direction,
        "magnitude":       round(magnitude, 2),
        "confidence":      round(confidence, 4),
        "current_price":   round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
    }

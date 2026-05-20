# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

from features import build_features, build_targets, make_sequences, FEATURE_COLS
from fetch import fetch_ohlcv

SEQ_LEN  = 60
SAVE_DIR = "saved_model"


def build_model(n_features):
    inp = Input(shape=(SEQ_LEN, n_features))

    x = LSTM(128, return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = LSTM(64, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation="relu")(x)
    x = Dropout(0.1)(x)

    direction_out = Dense(1, activation="sigmoid", name="direction")(x)
    magnitude_out = Dense(1, activation="linear",  name="magnitude")(x)

    model = Model(inputs=inp, outputs=[direction_out, magnitude_out])
    model.compile(
        optimizer=Adam(1e-3),
        loss={"direction": "binary_crossentropy", "magnitude": "huber"},
        loss_weights={"direction": 1.0, "magnitude": 0.5},
        metrics={"direction": "accuracy", "magnitude": "mae"},
    )
    return model


def train(ticker, start, end=None):
    print(f"Downloading {ticker} ...")
    df = fetch_ohlcv(ticker, start, end)

    print("Building features ...")
    df = build_features(df)
    direction, magnitude = build_targets(df, lookahead=1)

    df        = df.iloc[:-1]
    direction = direction.iloc[:-1].values
    magnitude = magnitude.iloc[:-1].values

    scaler = StandardScaler()
    feat   = scaler.fit_transform(df[FEATURE_COLS].values)

    X, y_dir, y_mag = make_sequences(feat, direction, magnitude, SEQ_LEN)
    print(f"Samples: {len(X)}  Features: {X.shape[2]}")

    X_tr, X_val, yd_tr, yd_val, ym_tr, ym_val = train_test_split(
        X, y_dir, y_mag, test_size=0.15, shuffle=False
    )

    model = build_model(X.shape[2])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
    ]

    print("Training ...")
    model.fit(
        X_tr,
        {"direction": yd_tr, "magnitude": ym_tr},
        validation_data=(X_val, {"direction": yd_val, "magnitude": ym_val}),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(os.path.join(SAVE_DIR, f"{ticker}_model.keras"))
    with open(os.path.join(SAVE_DIR, f"{ticker}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"Model saved to {SAVE_DIR}/")
    return model, scaler


def predict(ticker):
    model_path  = os.path.join(SAVE_DIR, f"{ticker}_model.keras")
    scaler_path = os.path.join(SAVE_DIR, f"{ticker}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found. Run: python predict.py train --ticker {ticker}"
        )

    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=SEQ_LEN + 300)).strftime("%Y-%m-%d")

    df   = fetch_ohlcv(ticker, start, end)
    df   = build_features(df)
    feat = scaler.transform(df[FEATURE_COLS].values)
    seq  = feat[-SEQ_LEN:][np.newaxis, :, :].astype(np.float32)

    dir_prob, mag_pred = model.predict(seq, verbose=0)
    dir_prob  = float(dir_prob[0][0])
    magnitude = float(mag_pred[0][0])

    current_price   = float(df["Close"].iloc[-1])
    predicted_price = current_price * (1 + magnitude / 100)
    direction       = "Up" if dir_prob >= 0.5 else "Down"
    confidence      = dir_prob if dir_prob >= 0.5 else 1 - dir_prob

    return {
        "direction":       direction,
        "magnitude":       round(magnitude, 2),
        "confidence":      round(confidence, 4),
        "current_price":   round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
    }

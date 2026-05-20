# 📈 StockSignal — 股票漲跌預測系統

> 預測股票**漲或跌**的方向，以及**漲跌幅度（%）**。

---

## 功能說明

本系統只做一件事：輸入股票代號，輸出：

- ✅ **方向**：漲 / 跌
- ✅ **幅度**：預測漲跌百分比（如 +3.2% / -1.8%）
- ✅ **信心分數**：模型對此次預測的把握程度（0~1）

---

## 專案結構

```
stock-predictor/
├── predict.py          # 主程式入口（CLI）
├── model.py            # LSTM 模型訓練與預測
├── features.py         # 技術指標特徵工程
├── fetch.py            # 下載股價資料
├── requirements.txt
└── README.md
```

---

## 安裝

```bash
git clone https://github.com/yourusername/stock-signal.git
cd stock-signal
pip install -r requirements.txt
```

---

## 使用方式

### 訓練模型

```bash
python predict.py train --ticker AAPL --start 2019-01-01
```

### 預測明日漲跌

```bash
python predict.py predict --ticker AAPL
```

### 輸出範例

```
📊 AAPL 明日預測
──────────────────
方向：▲ 上漲
幅度：+2.37%
信心：78.4%
預測價：$192.84（現價：$188.40）
```

---

## 模型說明

- 使用 **LSTM** 學習過去 60 天的價格序列與技術指標
- 輸出分成兩個 head：
  - **分類 head**：漲 / 跌（Binary Cross-Entropy）
  - **回歸 head**：漲跌幅度（Huber Loss）
- 特徵：RSI、MACD、Bollinger Band、成交量比、日報酬率

---

## ⚠️ 免責聲明

本工具僅供學習研究用途，**不構成任何投資建議**。股市有風險，請勿依賴此系統做實際交易決策。

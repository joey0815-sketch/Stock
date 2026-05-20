# -*- coding: utf-8 -*-
"""
Usage:
  python predict.py train   --ticker AAPL --start 2019-01-01
  python predict.py predict --ticker AAPL
"""
import argparse
import sys
from model import train, predict as run_predict


def cmd_train(args):
    train(ticker=args.ticker, start=args.start, end=args.end)


def cmd_predict(args):
    r     = run_predict(ticker=args.ticker)
    arrow = "UP  (+)" if r["direction"] == "Up" else "DOWN (-)"
    sign  = "+" if r["magnitude"] >= 0 else ""

    print(f"""
[{args.ticker}] Tomorrow's Prediction
{'─' * 34}
Direction  : {arrow}
Magnitude  : {sign}{r['magnitude']}%
Confidence : {r['confidence'] * 100:.1f}%
Current    : ${r['current_price']}
Predicted  : ${r['predicted_price']}
""")


def main():
    parser = argparse.ArgumentParser(description="StockSignal - Direction & Magnitude Predictor")
    sub    = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--ticker", required=True)
    p_train.add_argument("--start",  required=True)
    p_train.add_argument("--end",    default=None)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--ticker", required=True)

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

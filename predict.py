"""
predict.py — 主程式入口

用法：
  python predict.py train   --ticker AAPL --start 2019-01-01
  python predict.py predict --ticker AAPL
"""
import argparse
import sys
from model import train, predict as run_predict


def cmd_train(args):
    train(ticker=args.ticker, start=args.start, end=args.end)


def cmd_predict(args):
    result = run_predict(ticker=args.ticker)

    arrow  = "▲" if result["direction"] == "漲" else "▼"
    sign   = "+" if result["magnitude"] >= 0 else ""
    conf   = result["confidence"] * 100

    print(f"""
📊  {args.ticker} 明日預測
{'─' * 28}
方向：{arrow} {result['direction']}
幅度：{sign}{result['magnitude']}%
信心：{conf:.1f}%
現價：${result['current_price']}
預測：${result['predicted_price']}
""")


def main():
    parser = argparse.ArgumentParser(
        description="StockSignal — 股票漲跌預測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="訓練模型")
    p_train.add_argument("--ticker", required=True, help="股票代號，例如 AAPL")
    p_train.add_argument("--start",  required=True, help="訓練起始日 YYYY-MM-DD")
    p_train.add_argument("--end",    default=None,  help="訓練截止日（預設今天）")

    # predict
    p_pred = sub.add_parser("predict", help="預測明日漲跌")
    p_pred.add_argument("--ticker", required=True, help="股票代號，例如 AAPL")

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

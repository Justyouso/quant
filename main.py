#!/usr/bin/env python3
"""Quant - A股量化选股与回测系统"""

import argparse
import sys
from datetime import datetime

from config import DEFAULT_START, DEFAULT_END, INITIAL_CASH, TOP_N
from data_fetcher import DataFetcher
from strategies import BUILTIN_STRATEGIES
from backtest import BacktestEngine
from selector import StockSelector


def cmd_strategies(args):
    """列出内置策略"""
    print("可用的内置策略:\n")
    for name, cls in BUILTIN_STRATEGIES.items():
        print(f"  {name:20s}  {cls.__doc__ or ''}")
    print()


def cmd_backtest(args):
    """执行回测"""
    if args.strategy not in BUILTIN_STRATEGIES:
        print(f"❌ 未知策略: {args.strategy}")
        print(f"   可用策略: {', '.join(BUILTIN_STRATEGIES.keys())}")
        return

    codes = args.codes or None
    if not codes:
        print("📡 获取全部A股列表...")
        fetcher = DataFetcher()
        all_stocks = fetcher.get_all_stocks(args.end or datetime.now().strftime('%Y-%m-%d'))
        fetcher.close()
        codes = all_stocks['code'].tolist()[:50]  # 限制数量以免太慢
        print(f"   选取前 {len(codes)} 只股票进行回测")

    print(f"📡 下载 {len(codes)} 只股票数据...")
    fetcher = DataFetcher()
    data = fetcher.get_multi_daily(codes, args.start, args.end)
    fetcher.close()

    if not data:
        print("❌ 未获取到数据")
        return

    print(f"\n📊 运行回测 ({args.strategy}) ...")
    strategy_cls = BUILTIN_STRATEGIES[args.strategy]
    strategy = strategy_cls()
    engine = BacktestEngine(strategy, args.cash)
    result = engine.run(data)

    if 'error' in result:
        print(f"❌ {result['error']}")
        return

    # 输出结果
    print_result(result)


def print_result(r: dict):
    """格式化输出回测结果"""
    print("\n" + "=" * 50)
    print(f"  回测结果: {r['strategy']}")
    print("=" * 50)
    print(f"  区间:      {r['start_date']} → {r['end_date']}")
    print(f"  初始资金:  {r['initial_cash']:>12,.2f}")
    print(f"  最终资产:  {r['final_value']:>12,.2f}")
    print(f"  总收益率:  {r['total_return_pct']:>11.2f} %")
    print(f"  年化收益:  {r['annual_return_pct']:>11.2f} %")
    print(f"  夏普比率:  {r['sharpe_ratio']:>11.2f}")
    print(f"  最大回撤:  {r['max_drawdown_pct']:>11.2f} %")
    print(f"  总交易:    {r['total_trades']:>11}")
    print(f"  胜率:      {r['win_rate_pct']:>11.2f} %")
    print(f"  平均盈利:  {r['avg_win_pct']:>11.2f} %")
    print(f"  平均亏损:  {r['avg_loss_pct']:>11.2f} %")
    print(f"  持仓数:    {r['final_positions']:>11}")
    print("=" * 50)


def cmd_select(args):
    """执行选股"""
    if args.strategy not in BUILTIN_STRATEGIES:
        print(f"❌ 未知策略: {args.strategy}")
        print(f"   可用策略: {', '.join(BUILTIN_STRATEGIES.keys())}")
        return

    market = 'sh' if args.sh else 'sz' if args.sz else None
    scope = {'sh': '沪A', 'sz': '深A', None: '全市场'}[market]

    strategy_cls = BUILTIN_STRATEGIES[args.strategy]
    strategy = strategy_cls()

    selector = StockSelector(strategy)
    result = selector.select(top_n=args.top, verbose=True,
                             force_refresh=args.refresh, market=market)
    selector.close()

    if result.empty:
        print("\n❌ 未选出符合条件的股票")
        return

    print("\n" + "=" * 70)
    print(f"  选股结果 (策略: {args.strategy}, Top {len(result)})")
    print("=" * 70)
    print(result.to_string(index=False))
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Quant - A股量化选股与回测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 回测
  python main.py backtest --strategy golden_cross --codes sh.600000 sz.000001
  python main.py backtest --strategy macd --start 2023-01-01 --end 2025-12-31

  # 选股
  python main.py select --strategy value --top 15
  python main.py select --strategy golden_cross
  python main.py select --strategy golden_cross --sz   # 仅深A
  python main.py select --strategy golden_cross --sh   # 仅沪A

  # 查看策略
  python main.py strategies
        """
    )
    parser.add_argument('--verbose', action='store_true', default=True)

    sub = parser.add_subparsers(dest='command', required=True)

    # strategies
    sub.add_parser('strategies', help='列出所有内置策略')

    # backtest
    bt = sub.add_parser('backtest', help='回测策略')
    bt.add_argument('--strategy', required=True, help='策略名称')
    bt.add_argument('--codes', nargs='+', default=None, help='股票代码列表')
    bt.add_argument('--start', default=DEFAULT_START, help=f'开始日期 (默认 {DEFAULT_START})')
    bt.add_argument('--end', default=DEFAULT_END, help=f'结束日期 (默认 {DEFAULT_END})')
    bt.add_argument('--cash', type=float, default=INITIAL_CASH, help=f'初始资金 (默认 {INITIAL_CASH})')

    # select
    sl = sub.add_parser('select', help='实时选股')
    sl.add_argument('--strategy', default='value', help='策略名称 (默认 value)')
    sl.add_argument('--top', type=int, default=TOP_N, help=f'输出前N只 (默认 {TOP_N})')
    sl.add_argument('--refresh', action='store_true', help='强制重新下载数据，忽略缓存')
    group = sl.add_mutually_exclusive_group()
    group.add_argument('--sz', action='store_true', help='仅选深A股票')
    group.add_argument('--sh', action='store_true', help='仅选沪A股票')

    args = parser.parse_args()

    if args.command == 'strategies':
        cmd_strategies(args)
    elif args.command == 'backtest':
        cmd_backtest(args)
    elif args.command == 'select':
        cmd_select(args)


if __name__ == '__main__':
    main()

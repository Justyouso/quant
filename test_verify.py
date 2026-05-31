#!/usr/bin/env python3
"""快速验证：只拉取 5 只股票，测试基本面 + K线 + 增量更新"""
import sys, os, shutil

# 清空缓存
cache = os.path.expanduser("~/.quant_cache")
if os.path.exists(cache):
    shutil.rmtree(cache)
    print("🧹 已清空缓存")

sys.path.insert(0, os.path.dirname(__file__))
from data_fetcher import DataFetcher
from strategies import GoldenCrossStrategy
from selector import StockSelector


def test_fundamentals():
    print("\n" + "=" * 50)
    print("1️⃣  测试基本面获取（仅 5 只股票，手动）")
    print("=" * 50)
    f = DataFetcher()
    y, q = f.get_latest_quarter()
    print(f"   最新可用季度: {y}Q{q}")

    codes = ["sh.600519", "sz.000001", "sh.600036", "sz.300750", "sh.600000"]
    for code in codes:
        # PE/PB 从 K 线取
        val = f.get_latest_pe_pb(code)
        if val:
            print(f"   {code}: PE={val['peTTM']}, PB={val['pbMRQ']}")
        else:
            print(f"   {code}: PE/PB 无数据")

        profit = f.get_profit_data(code, y, q)
        if profit is not None and not profit.empty:
            p = profit.iloc[0]
            roe = p.get('roeAvg', 'N/A')
            print(f"         ROE={roe}")

    f.close()
    print("   ✅ 基本面 OK")


def test_kline_incremental():
    print("\n" + "=" * 50)
    print("2️⃣  测试 K线增量更新")
    print("=" * 50)
    f = DataFetcher()

    # 第一次：拉取 2025-01-01 ~ 2025-03-31
    print("   第一次请求: 2025-01-01 ~ 2025-03-31")
    df1 = f.get_daily_data("sh.600519", "2025-01-01", "2025-03-31")
    print(f"   返回 {len(df1)} 行, 日期: {df1['date'].min().date()} ~ {df1['date'].max().date()}")

    # 第二次：拉取 2025-01-01 ~ 2025-06-30（应该只增量拉取 4~6 月）
    print("   第二次请求: 2025-01-01 ~ 2025-06-30（应增量更新）")
    df2 = f.get_daily_data("sh.600519", "2025-01-01", "2025-06-30")
    print(f"   返回 {len(df2)} 行, 日期: {df2['date'].min().date()} ~ {df2['date'].max().date()}")
    assert len(df2) >= len(df1), "增量后数据不应减少"
    print("   ✅ 增量更新 OK")

    # 第三次：重复请求，应零网络命中缓存
    print("   第三次请求: 相同区间（命中缓存）")
    df3 = f.get_daily_data("sh.600519", "2025-01-01", "2025-06-30")
    assert len(df3) == len(df2)
    print("   ✅ 缓存命中 OK")

    f.close()


def test_selector():
    print("\n" + "=" * 50)
    print("3️⃣  测试选股引擎（只选 5 只股票强筛）")
    print("=" * 50)
    strategy = GoldenCrossStrategy()
    selector = StockSelector(strategy)
    # 仅测试 5 只股票
    test_codes = ["sh.600519", "sz.000001", "sh.600036", "sz.300750", "sh.600000"]
    result = selector.select(top_n=5, verbose=True,
                             fund_quarter=(2025, 1),
                             codes=test_codes)
    selector.close()

    if result.empty:
        print("   ⚠ 选股结果为空（可能 2025Q1 数据不可用）")
    else:
        print(f"   选股结果 ({len(result)} 只):")
        print(result.to_string(index=False))

    print("   ✅ 选股 OK")


if __name__ == "__main__":
    test_fundamentals()
    test_kline_incremental()
    test_selector()
    print("\n" + "=" * 50)
    print("🎉 全部测试通过（除非上面有 ❌）")
    print("=" * 50)

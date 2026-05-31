"""实时选股引擎"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from indicators import fundamental_filters
from config import TOP_N, MIN_DAYS


class StockSelector:
    """选股引擎：基本面初筛 → 技术面打分"""

    def __init__(self, strategy):
        self.strategy = strategy
        self.fetcher = DataFetcher()

    def select(self, top_n: int = TOP_N, days: int = MIN_DAYS,
               fund_quarter: tuple = None, verbose: bool = True,
               force_refresh: bool = False,
               codes: list = None,
               market: str = None) -> pd.DataFrame:
        """执行选股

        Args:
            top_n: 输出前 N 只
            days: 最近 N 天数据
            fund_quarter: (year, quarter)，None 则自动选择最新
            codes: 限定股票代码列表，None 则全市场
            market: 'sh'=沪A, 'sz'=深A, None=全市场
        """
        if fund_quarter is None:
            fund_quarter = self.fetcher.get_latest_quarter()

        year, quarter = fund_quarter
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=days + 60)).strftime('%Y-%m-%d')

        if verbose:
            scope = f"{market.upper()} " if market else "全市场"
            print(f"📡 获取 {year}Q{quarter} {scope}基本面数据...")

        fund_df = self.fetcher.get_all_fundamentals(year, quarter, verbose=verbose,
                                                    force_refresh=force_refresh,
                                                    codes=codes)
        if fund_df.empty:
            print("⚠ 未获取到基本面数据")
            return pd.DataFrame()

        # 基本面初筛
        fund_df = fundamental_filters(fund_df)
        screened = fund_df[
            fund_df.get('filter_pe', True) &
            fund_df.get('filter_pb', True) &
            fund_df.get('filter_roe', True)
        ].copy()

        if verbose:
            print(f"  基本面初筛通过: {len(screened)}/{len(fund_df)} 只")

        if screened.empty:
            print("⚠ 无股票通过基本面筛选")
            return pd.DataFrame()

        # 获取日线数据计算技术信号
        codes = screened['code'].tolist()

        # 市场过滤
        if market == 'sh':
            codes = [c for c in codes if c.startswith('sh.')]
        elif market == 'sz':
            codes = [c for c in codes if c.startswith('sz.')]

        if verbose:
            print(f"📡 下载 {len(codes)} 只股票日K线...")

        daily_data = self.fetcher.get_multi_daily(codes, start, end, verbose=verbose)

        # 逐只打分
        results = []
        for code in codes:
            df = daily_data.get(code)
            if df is None or len(df) < MIN_DAYS:
                continue

            df = self.strategy.compute_indicators(df)
            latest = df.iloc[-1]

            # 信号强度
            score_series = self.strategy.score(df)
            tech_score = score_series.iloc[-1] if not score_series.empty else 0.0

            # 基础检查
            if pd.isna(latest.get('close')) or latest['close'] <= 0:
                continue
            if latest.get('turn', 100) < 0.1:  # 排除几乎无成交的
                continue

            # 基本信息
            fund_row = screened[screened['code'] == code]
            pe = fund_row['peTTM'].values[0] if 'peTTM' in fund_row else None
            roe = fund_row['roeAvg'].values[0] if 'roeAvg' in fund_row else None

            # 综合评分 = 基本面分数 + 技术面分数
            fund_score = fund_row['fundamental_score'].values[0] if 'fundamental_score' in fund_row else 0
            combined = fund_score * 0.4 + tech_score * 0.6

            # 市场标识
            exchange = '沪A' if code.startswith('sh.') else '深A'

            results.append({
                'code': code,
                'market': exchange,
                'close': latest['close'],
                'pctChg': latest.get('pctChg', 0),
                'turn': latest.get('turn', 0),
                'pe': round(pe, 2) if pe and not np.isnan(pe) else None,
                'roe': round(roe, 2) if roe and not np.isnan(roe) else None,
                'fund_score': round(fund_score, 3),
                'tech_score': round(tech_score, 3),
                'combined_score': round(combined, 3),
            })

        result_df = pd.DataFrame(results)
        if result_df.empty:
            print("⚠ 无符合条件的股票")
            return result_df

        result_df = result_df.sort_values('combined_score', ascending=False).head(top_n)
        result_df['rank'] = range(1, len(result_df) + 1)
        cols = ['rank', 'code', 'market', 'close', 'pctChg', 'turn', 'pe', 'roe',
                'fund_score', 'tech_score', 'combined_score']
        return result_df[cols]

    def close(self):
        self.fetcher.close()

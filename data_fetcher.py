"""baostock 数据接口封装"""

import os
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional
from config import DEFAULT_START, DEFAULT_END

CACHE_DIR = os.path.expanduser("~/.quant_cache")


class DataFetcher:
    """A股数据获取，基于 baostock，带本地文件缓存"""

    def __init__(self):
        self._logged_in = False
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _ensure_login(self):
        if not self._logged_in:
            lg = bs.login()
            if lg.error_code != '0':
                raise ConnectionError(f"登录 baostock 失败: {lg.error_msg}")
            self._logged_in = True

    def close(self):
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    def __enter__(self):
        self._ensure_login()
        return self

    def __exit__(self, *args):
        self.close()

    # ─── 缓存工具 ───────────────────────────────────

    @staticmethod
    def _cache_path(name: str) -> str:
        return os.path.join(CACHE_DIR, name)

    @staticmethod
    def _load_cache(name: str) -> Optional[pd.DataFrame]:
        path = DataFetcher._cache_path(name)
        if os.path.exists(path):
            try:
                return pd.read_csv(path, parse_dates=['date'])
            except Exception:
                return None
        return None

    @staticmethod
    def _save_cache(df: pd.DataFrame, name: str):
        path = DataFetcher._cache_path(name)
        df.to_csv(path, index=False)

    # ─── 股票列表 ────────────────────────────────

    def get_all_stocks(self, date: str) -> pd.DataFrame:
        """获取指定日期全部正常上市的A股"""
        self._ensure_login()
        rs = bs.query_all_stock(date)
        df = rs.get_data()
        if df.empty:
            return df
        # 只保留 A 股（上海主板/科创板、深圳主板/中小板/创业板）
        df = df[df['code'].str.match(r'(sh\.6\d{4}|sh\.688\d{3}|sz\.0\d{4}|sz\.002\d{3}|sz\.300\d{3})')]
        return df[['code', 'tradeStatus']].copy()

    def get_stock_codes_in_date_range(self, start: str, end: str) -> list:
        """获取日期范围内上市的全部股票代码"""
        seen = set()
        codes = []
        for y in range(int(start[:4]), int(end[:4]) + 1):
            df = self.get_all_stocks(f"{y}-12-31" if y < int(end[:4]) else end)
            for c in df['code']:
                if c not in seen:
                    seen.add(c)
                    codes.append(c)
        return codes

    # ─── K线数据 ────────────────────────────────

    def get_daily_data(self, code: str, start: str = DEFAULT_START,
                       end: str = DEFAULT_END, adjustflag: str = '2',
                       use_cache: bool = True) -> pd.DataFrame:
        """获取日K线数据，支持增量更新

        adjustflag: '1'=不复权, '2'=前复权, '3'=后复权
        use_cache: 启用本地缓存。每只股票一个独立 parquet 文件，
                   自动增量拉取缺失区间。
        """
        stock_cache = f"kline_{code}_adj{adjustflag}.parquet"
        need_start = pd.Timestamp(start)
        need_end = pd.Timestamp(end)

        # ── 读取本地缓存 ────────────────────────────
        cached_df = None
        if use_cache:
            cached_df = self._load_cache(stock_cache)

        if cached_df is not None and not cached_df.empty:
            cache_start = cached_df['date'].min()
            cache_end = cached_df['date'].max()

            # 情况 1: 缓存已全覆盖请求区间
            if cache_start <= need_start and cache_end >= need_end:
                return cached_df[
                    (cached_df['date'] >= need_start) &
                    (cached_df['date'] <= need_end)
                ].copy()

            inc_start, inc_end = None, None
            # 情况 2: 尾部缺失 → 仅拉取缺失尾部
            if cache_end < need_end:
                inc_start = (cache_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                inc_end = end
            # 情况 3: 头部缺失 → 仅拉取缺失头部
            if cache_start > need_start:
                if inc_start is None:
                    inc_start = start
                    inc_end = (cache_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    # 头部尾部都缺失 → 拉完整区间替换缓存
                    inc_start, inc_end = start, end
        else:
            # 无缓存 → 全量拉取
            inc_start, inc_end = start, end

        # ── 向 baostock 请求 ──────────────────────
        if inc_start is not None:
            self._ensure_login()
            fields = 'date,open,high,low,close,preclose,volume,amount,adjustflag,turn,pctChg'
            rs = bs.query_history_k_data_plus(
                code, fields, start_date=inc_start, end_date=inc_end,
                frequency='d', adjustflag=adjustflag
            )
            new_df = rs.get_data()
            if new_df.empty:
                if cached_df is not None and not cached_df.empty:
                    return cached_df[
                        (cached_df['date'] >= need_start) &
                        (cached_df['date'] <= need_end)
                    ].copy()
                return new_df

            numeric_cols = ['open', 'high', 'low', 'close', 'preclose',
                            'volume', 'amount', 'turn', 'pctChg']
            for c in numeric_cols:
                new_df[c] = pd.to_numeric(new_df[c], errors='coerce')
            new_df['date'] = pd.to_datetime(new_df['date'])
            new_df['code'] = code

            # ── 合并 & 缓存 ──
            if cached_df is not None and not cached_df.empty:
                merged = pd.concat([cached_df, new_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
            else:
                merged = new_df.sort_values('date').reset_index(drop=True)

            if use_cache:
                self._save_cache(merged, stock_cache)

            return merged[
                (merged['date'] >= need_start) &
                (merged['date'] <= need_end)
            ].copy()
        else:
            # 缓存已全覆盖（仅头部缺失的单向场景已被上方短路）
            return cached_df[
                (cached_df['date'] >= need_start) &
                (cached_df['date'] <= need_end)
            ].copy()

    def get_multi_daily(self, codes: list, start: str = DEFAULT_START,
                        end: str = DEFAULT_END, adjustflag: str = '2',
                        verbose: bool = True, use_cache: bool = True) -> dict:
        """批量获取多只股票日K线，返回 {code: DataFrame}"""
        result = {}
        for i, code in enumerate(codes):
            if verbose:
                print(f"\r  下载 {code}  ({i + 1}/{len(codes)})", end='', flush=True)
            try:
                df = self.get_daily_data(code, start, end, adjustflag,
                                         use_cache=use_cache)
                if not df.empty:
                    result[code] = df
            except Exception as e:
                if verbose:
                    print(f"\n  ⚠ {code}: {e}")
        if verbose:
            print()
        return result

    # ─── 基本面数据 ──────────────────────────────

    @staticmethod
    def _nearest_trading_day(year: int, month: int, day: int = 15) -> str:
        """寻找最近的可能交易日"""
        d = date(year, month, day)
        for offset in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -7, -10, -14]:
            dt = d + timedelta(days=offset)
            if dt.weekday() < 5:
                return dt.strftime('%Y-%m-%d')
        return d.strftime('%Y-%m-%d')

    def get_latest_quarter(self) -> tuple:
        """获取最新可用的财报季度，并验证数据是否可用"""
        self._ensure_login()
        today = datetime.now()
        m = today.month
        candidates = []
        if m <= 4:
            candidates = [(today.year - 1, 4), (today.year - 1, 3)]
        elif m <= 8:
            candidates = [(today.year, 1), (today.year - 1, 4)]
        elif m <= 10:
            candidates = [(today.year, 2), (today.year, 1)]
        else:
            candidates = [(today.year, 3), (today.year, 2)]

        for year, quarter in candidates:
            try:
                rs = bs.query_profit_data("sh.600000", year, quarter)
                if not rs.get_data().empty:
                    return year, quarter
            except Exception:
                continue
        return candidates[0]

    def get_stock_basic(self, code: str) -> Optional[pd.DataFrame]:
        """获取股票基本信息（含 peTTM, pbMRQ, ROE）"""
        self._ensure_login()
        rs = bs.query_stock_basic(code)
        data = rs.get_data()
        return data if not data.empty else None

    def get_profit_data(self, code: str, year: int, quarter: int) -> Optional[pd.DataFrame]:
        """盈利能力数据：每股收益、ROE等"""
        self._ensure_login()
        rs = bs.query_profit_data(code, year, quarter)
        data = rs.get_data()
        return data if not data.empty else None

    def get_growth_data(self, code: str, year: int, quarter: int) -> Optional[pd.DataFrame]:
        """成长能力数据：营收/利润增长率"""
        self._ensure_login()
        rs = bs.query_growth_data(code, year, quarter)
        data = rs.get_data()
        return data if not data.empty else None

    def get_latest_pe_pb(self, code: str, trade_date: str = None) -> Optional[dict]:
        """通过日K线获取最新 PE / PB"""
        self._ensure_login()
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y-%m-%d')
        # 若当天非交易日，往前试几天
        candidates = [trade_date]
        d = datetime.strptime(trade_date, '%Y-%m-%d')
        for offset in [-1, -2, -3, -4, -5, -7, -10]:
            dt = d + timedelta(days=offset)
            candidates.append(dt.strftime('%Y-%m-%d'))

        for dt_str in candidates:
            rs = bs.query_history_k_data_plus(
                code,
                "peTTM,pbMRQ",
                start_date=dt_str, end_date=dt_str,
                frequency='d', adjustflag='2'
            )
            data = rs.get_data()
            if not data.empty:
                row = data.iloc[0]
                pe = float(row['peTTM']) if 'peTTM' in row and row['peTTM'] != '' else None
                pb = float(row['pbMRQ']) if 'pbMRQ' in row and row['pbMRQ'] != '' else None
                if pe is not None or pb is not None:
                    return {'peTTM': pe, 'pbMRQ': pb}
        return None

    def get_all_fundamentals(self, year: int, quarter: int,
                             verbose: bool = True, use_cache: bool = True,
                             force_refresh: bool = False,
                             codes: list = None) -> pd.DataFrame:
        """批量获取全市场基本面数据，带本地缓存

        PE/PB 从日K线取（query_stock_basic 在 0.9.x 无此字段）。
        ROE/增长率从季报取。
        """
        cache_name = f"fundamentals_{year}Q{quarter}.csv"
        is_subset = codes is not None
        orig_codes = codes

        # ── 读缓存（优先全量，子集直接从全量缓存过滤） ──
        if use_cache and not force_refresh:
            full_cache = self._load_cache(cache_name)
            if full_cache is not None and not full_cache.empty:
                if is_subset:
                    filtered = full_cache[full_cache['code'].isin(orig_codes)]
                    if verbose:
                        print(f"  → 从缓存过滤 ({len(filtered)} 只)")
                    return filtered
                if verbose:
                    print(f"  → 命中缓存: {self._cache_path(cache_name)} ({len(full_cache)} 只)")
                return full_cache

        # 无论是否子集，都全量拉取建缓存（子集从缓存过滤）
        today_str = datetime.now().strftime('%Y-%m-%d')
        stocks = self.get_all_stocks(today_str)
        if stocks.empty:
            stocks = self.get_all_stocks(
                self._nearest_trading_day(year, quarter * 3, 15))
        if stocks.empty:
            return pd.DataFrame()
        all_codes = stocks['code'].tolist()
        rows = []

        for i, code in enumerate(all_codes):
            if verbose:
                print(f"\r  基本面 {i + 1}/{len(all_codes)}", end='', flush=True)
            try:
                # PE/PB 从日K线取
                val = self.get_latest_pe_pb(code)
                pe = val['peTTM'] if val else None
                pb = val['pbMRQ'] if val else None

                if pe is not None and (pe <= 0 or pe > 200):
                    continue
                if pb is not None and pb <= 0:
                    continue

                row = {'code': code, 'peTTM': pe, 'pbMRQ': pb,
                       'roeAvg': None, 'epsTTM': None}

                profit = self.get_profit_data(code, year, quarter)
                if profit is not None and not profit.empty:
                    p = profit.iloc[0]
                    if 'roeAvg' in p and p['roeAvg'] != '':
                        row['roeAvg'] = float(p['roeAvg'])
                    if 'epsTTM' in p and p['epsTTM'] != '':
                        row['epsTTM'] = float(p['epsTTM'])
                    if 'profitYOY' in p and p['profitYOY'] != '':
                        row['profitYOY'] = float(p['profitYOY'])

                growth = self.get_growth_data(code, year, quarter)
                if growth is not None and not growth.empty:
                    g = growth.iloc[0]
                    for col in ['YOYIncome', 'YOYProfit', 'YOYEquity']:
                        if col in g and g[col] != '':
                            row[col] = float(g[col])

                rows.append(row)
            except Exception:
                continue

        if verbose:
            print()

        result = pd.DataFrame(rows)

        # ── 写入缓存（始终存全量） ──
        if use_cache and not result.empty:
            self._save_cache(result, cache_name)
            if verbose:
                print(f"  → 已缓存: {self._cache_path(cache_name)} ({len(result)} 只)")

        # ── 子集请求则过滤返回 ──
        if is_subset and not result.empty:
            result = result[result['code'].isin(orig_codes)]

        return result

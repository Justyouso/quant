"""技术面 + 基本面指标计算"""

import pandas as pd
import numpy as np
from config import (
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_PERIOD,
    KDJ_K, KDJ_D, BOLL_PERIOD, BOLL_STD,
)


# ═══════════════════════════════════════════════
# 技术指标
# ═══════════════════════════════════════════════

def add_ma(df: pd.DataFrame, periods=None) -> pd.DataFrame:
    """移动平均线"""
    periods = periods or [5, 10, 20, 60, 120, 250]
    for p in periods:
        df[f'MA{p}'] = df['close'].rolling(p).mean()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD 指标"""
    fast, slow, signal = MACD_FAST, MACD_SLOW, MACD_SIGNAL

    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['DIF'] = ema_fast - ema_slow
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    return df


def add_rsi(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
    """RSI 指标"""
    period = period or RSI_PERIOD
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def add_kdj(df: pd.DataFrame) -> pd.DataFrame:
    """KDJ 指标"""
    k_period = KDJ_K
    low_n = df['low'].rolling(k_period).min()
    high_n = df['high'].rolling(k_period).max()
    rsv = ((df['close'] - low_n) / (high_n - low_n).replace(0, np.nan)) * 100

    df['KDJ_K'] = rsv.ewm(com=KDJ_D - 1, adjust=False).mean()
    df['KDJ_D'] = df['KDJ_K'].ewm(com=KDJ_D - 1, adjust=False).mean()
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']
    return df


def add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    """布林带"""
    period, std = BOLL_PERIOD, BOLL_STD
    df['BOLL_MID'] = df['close'].rolling(period).mean()
    rolling_std = df['close'].rolling(period).std()
    df['BOLL_UP'] = df['BOLL_MID'] + std * rolling_std
    df['BOLL_DN'] = df['BOLL_MID'] - std * rolling_std
    df['BOLL_WIDTH'] = df['BOLL_UP'] - df['BOLL_DN']
    df['BOLL_POS'] = ((df['close'] - df['BOLL_DN']) /
                      df['BOLL_WIDTH'].replace(0, np.nan))
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """成交量指标"""
    df['VOL_MA5'] = df['volume'].rolling(5).mean()
    df['VOL_MA20'] = df['volume'].rolling(20).mean()
    df['VOL_RATIO'] = df['volume'] / df['VOL_MA5'].replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR (平均真实波幅)"""
    high_low = df['high'] - df['low']
    high_pc = (df['high'] - df['close'].shift(1)).abs()
    low_pc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(period).mean()
    return df


def add_all_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """添加全部常用技术指标"""
    df = df.copy()
    df = add_ma(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_kdj(df)
    df = add_bollinger(df)
    df = add_volume_indicators(df)
    df = add_atr(df)
    return df


# ═══════════════════════════════════════════════
# 基本面筛选（配合选股使用）
# ═══════════════════════════════════════════════

def fundamental_filters(fund_df: pd.DataFrame) -> pd.DataFrame:
    """添加基本面辅助列：根据常见选股条件标记"""
    df = fund_df.copy()

    # PE (TTM) 为正且合理
    df['filter_pe'] = (df.get('peTTM', pd.Series(0)) > 0) & (df.get('peTTM', pd.Series(80)) < 80)

    # PB > 0
    df['filter_pb'] = df.get('pbMRQ', pd.Series(1)) > 0

    # ROE 为正
    df['filter_roe'] = df.get('roeAvg', pd.Series(0)) > 0

    # 营收正增长
    df['filter_growth'] = df.get('YOYIncome', pd.Series(0)) > 0

    # 综合
    df['fundamental_score'] = 0.0
    for col, weight in [('peTTM', -0.3), ('roeAvg', 0.3),
                        ('YOYIncome', 0.2), ('YOYProfit', 0.2)]:
        if col in df and df[col].notna().sum() > 0:
            series = df[col].replace([np.inf, -np.inf], np.nan)
            min_v, max_v = series.quantile(0.01), series.quantile(0.99)
            scale = max_v - min_v
            if scale > 0:
                norm = (series - min_v) / scale
            else:
                norm = pd.Series(0.5, index=series.index)
            norm = norm.clip(0, 1)
            df['fundamental_score'] += weight * norm

    return df

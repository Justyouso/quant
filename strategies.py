"""策略框架与内置策略"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from indicators import add_all_technicals, fundamental_filters


class Strategy(ABC):
    """策略基类"""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """在 DataFrame 上计算指标并添加列"""

    @abstractmethod
    def buy_signal(self, df: pd.DataFrame) -> pd.Series:
        """返回 bool Series，True=买入"""

    @abstractmethod
    def sell_signal(self, df: pd.DataFrame) -> pd.Series:
        """返回 bool Series，True=卖出"""

    def score(self, df: pd.DataFrame) -> pd.Series:
        """信号强度评分（选股用），默认返回 0~1 分值"""
        return pd.Series(0.0, index=df.index)

    def __repr__(self):
        return f"<Strategy: {self.name}>"


# ═══════════════════════════════════════════════
# 内置策略
# ═══════════════════════════════════════════════

class GoldenCrossStrategy(Strategy):
    """双均线金叉死叉策略 (MA5 × MA20)"""

    def __init__(self, fast=5, slow=20):
        super().__init__(name=f"GoldenCross({fast},{slow})")
        self.fast = fast
        self.slow = slow

    def compute_indicators(self, df):
        df = df.copy()
        df[f'MA{self.fast}'] = df['close'].rolling(self.fast).mean()
        df[f'MA{self.slow}'] = df['close'].rolling(self.slow).mean()
        return df

    def buy_signal(self, df):
        fast = f'MA{self.fast}'
        slow = f'MA{self.slow}'
        return (df[fast] > df[slow]) & (df[fast].shift(1) <= df[slow].shift(1))

    def sell_signal(self, df):
        fast = f'MA{self.fast}'
        slow = f'MA{self.slow}'
        return (df[fast] < df[slow]) & (df[fast].shift(1) >= df[slow].shift(1))

    def score(self, df):
        fast = f'MA{self.fast}'
        slow = f'MA{self.slow}'
        ratio = df[fast] / df[slow]
        score = (ratio - 1).clip(0, 0.2) / 0.2
        return score.fillna(0)


class MACDStrategy(Strategy):
    """MACD 金叉死叉策略"""

    def compute_indicators(self, df):
        from indicators import add_macd
        return add_macd(df.copy())

    def buy_signal(self, df):
        return (df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1))

    def sell_signal(self, df):
        return (df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1))

    def score(self, df):
        spread = (df['DIF'] - df['DEA']).abs()
        score = spread / spread.rolling(60).max().replace(0, np.nan)
        return score.fillna(0)


class RSIStrategy(Strategy):
    """RSI 超卖超买策略"""

    def __init__(self, period=14, oversold=30, overbought=70):
        super().__init__(name=f"RSI({period})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_indicators(self, df):
        from indicators import add_rsi
        return add_rsi(df.copy(), self.period)

    def buy_signal(self, df):
        return (df['RSI'] < self.oversold) & (df['RSI'].shift(1) >= self.oversold)

    def sell_signal(self, df):
        return (df['RSI'] > self.overbought) & (df['RSI'].shift(1) <= self.overbought)

    def score(self, df):
        score = (self.oversold - df['RSI']).clip(0, self.oversold) / self.oversold
        return score.fillna(0)


class BollingerStrategy(Strategy):
    """布林带突破策略"""

    def compute_indicators(self, df):
        from indicators import add_bollinger
        return add_bollinger(df.copy())

    def buy_signal(self, df):
        return (df['close'] > df['BOLL_UP']) & (df['close'].shift(1) <= df['BOLL_UP'].shift(1))

    def sell_signal(self, df):
        return (df['close'] < df['BOLL_DN']) & (df['close'].shift(1) >= df['BOLL_DN'].shift(1))

    def score(self, df):
        # 布林带位置：越靠近下轨买入信号越强
        pos = df['BOLL_POS'].fillna(0.5)
        score = (0.5 - pos).clip(0, 0.5) / 0.5
        return score.fillna(0)


class ValueStrategy(Strategy):
    """基本面选股策略（仅供选股，不支持回测）"""

    def __init__(self, max_pe=30, min_roe=5):
        super().__init__("Value")
        self.max_pe = max_pe
        self.min_roe = min_roe

    def compute_indicators(self, df):
        return df

    def buy_signal(self, df):
        return pd.Series(False, index=df.index)

    def sell_signal(self, df):
        return pd.Series(False, index=df.index)

    def score_fundamentals(self, fund_df: pd.DataFrame) -> pd.DataFrame:
        """对全市场基本面打分"""
        df = fundamental_filters(fund_df)
        return df.sort_values('fundamental_score', ascending=False)


class CompositeStrategy(Strategy):
    """复合策略：综合多个子策略信号"""

    def __init__(self, strategies: list, weights: list = None):
        super().__init__("Composite")
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)

    def compute_indicators(self, df):
        df = df.copy()
        for s in self.strategies:
            df = s.compute_indicators(df)
        return df

    def buy_signal(self, df):
        if not self.strategies:
            return pd.Series(False, index=df.index)
        signals = [s.buy_signal(df).astype(float) * w
                   for s, w in zip(self.strategies, self.weights)]
        total = pd.concat(signals, axis=1).sum(axis=1)
        threshold = sum(w for w in self.weights if w > 0) * 0.5
        return total >= threshold

    def sell_signal(self, df):
        if not self.strategies:
            return pd.Series(False, index=df.index)
        signals = [s.sell_signal(df).astype(float) * w
                   for s, w in zip(self.strategies, self.weights)]
        total = pd.concat(signals, axis=1).sum(axis=1)
        threshold = sum(w for w in self.weights if w > 0) * 0.5
        return total >= threshold

    def score(self, df):
        scores = [s.score(df) * w for s, w in zip(self.strategies, self.weights)]
        return pd.concat(scores, axis=1).sum(axis=1).fillna(0)


# 策略注册表
BUILTIN_STRATEGIES = {
    'golden_cross': GoldenCrossStrategy,
    'macd': MACDStrategy,
    'rsi': RSIStrategy,
    'bollinger': BollingerStrategy,
    'value': ValueStrategy,
}

"""回测引擎"""

import pandas as pd
import numpy as np
from collections import defaultdict
from config import (
    INITIAL_CASH, MAX_POSITION_PCT, MAX_CONCURRENT,
    TRADE_SLIPPAGE, COMMISSION_RATE, STAMP_TAX_RATE, MIN_COMMISSION,
)


class Position:
    """持仓记录"""

    def __init__(self, code, buy_date, buy_price, shares, commission=0):
        self.code = code
        self.buy_date = buy_date
        self.buy_price = buy_price
        self.shares = shares
        self.cost = buy_price * shares
        self.commission = commission
        self.current_price = buy_price

    @property
    def market_value(self):
        return self.current_price * self.shares

    @property
    def pnl(self):
        return self.market_value - self.cost

    @property
    def pnl_pct(self):
        return self.pnl / self.cost if self.cost > 0 else 0.0


class Trade:
    """成交记录"""

    def __init__(self, code, date, direction, price, shares, commission, tax):
        self.code = code
        self.date = date
        self.direction = direction  # 'buy' or 'sell'
        self.price = price
        self.shares = shares
        self.amount = price * shares
        self.commission = commission
        self.tax = tax


class BacktestEngine:
    """回测引擎"""

    def __init__(self, strategy, initial_cash: float = INITIAL_CASH):
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []
        self._current_date = None

    def _calc_commission(self, amount: float) -> float:
        return max(amount * COMMISSION_RATE, MIN_COMMISSION)

    def _calc_sell_tax(self, amount: float) -> float:
        return amount * STAMP_TAX_RATE

    def _open_position(self, code: str, date, price: float):
        max_amount = self.cash * MAX_POSITION_PCT
        if max_amount <= 0:
            return
        shares = int(max_amount / (price * (1 + TRADE_SLIPPAGE))) // 100 * 100
        if shares < 100:
            return
        actual_price = price * (1 + TRADE_SLIPPAGE)
        amount = actual_price * shares
        commission = self._calc_commission(amount)
        total_cost = amount + commission

        if total_cost > self.cash:
            shares = int((self.cash - commission) / actual_price) // 100 * 100
            if shares < 100:
                return
            amount = actual_price * shares
            commission = self._calc_commission(amount)
            total_cost = amount + commission

        self.cash -= total_cost
        self.positions[code] = Position(code, date, actual_price, shares, commission)
        self.trades.append(Trade(code, date, 'buy', actual_price, shares, commission, 0))

    def _close_position(self, code: str, date, price: float):
        pos = self.positions.get(code)
        if not pos:
            return
        actual_price = price * (1 - TRADE_SLIPPAGE)
        amount = actual_price * pos.shares
        commission = self._calc_commission(amount)
        tax = self._calc_sell_tax(amount)
        net = amount - commission - tax

        self.cash += net
        self.trades.append(Trade(code, date, 'sell', actual_price, pos.shares, commission, tax))
        del self.positions[code]

    def run(self, data: dict[str, pd.DataFrame]) -> dict:
        """运行回测

        Args:
            data: {code: DataFrame}, 每只股票的数据需包含 open, close, date 列
        Returns:
            回测结果 dict
        """
        self.reset()

        # 计算所有股票的技术指标
        prepared = {}
        for code, df in data.items():
            if df.empty:
                continue
            df = df.sort_values('date').reset_index(drop=True)
            df = self.strategy.compute_indicators(df)
            prepared[code] = df

        # 收集所有交易日并排序
        all_dates = set()
        for df in prepared.values():
            all_dates.update(df['date'].dt.date.unique())
        all_dates = sorted(all_dates)

        # 逐日模拟
        for date in all_dates:
            self._current_date = date
            date_str = str(date)
            daily_value = self.cash
            buy_today = {}
            sell_today = {}

            for code, df in prepared.items():
                day_data = df[df['date'].dt.date == date]
                if day_data.empty:
                    continue
                row = day_data.iloc[-1]
                price = row['open']  # 以开盘价执行
                if pd.isna(price) or price <= 0:
                    continue

                # 更新持仓市值
                if code in self.positions:
                    self.positions[code].current_price = price
                    daily_value += self.positions[code].market_value

                # 获取信号
                idx = row.name
                buy = self.strategy.buy_signal(df)
                sell = self.strategy.sell_signal(df)

                if idx in buy.index and buy.loc[idx] and code not in self.positions and price > 0:
                    buy_today[code] = price
                if idx in sell.index and sell.loc[idx] and code in self.positions:
                    sell_today[code] = price

            # 先卖后买
            for code, price in sell_today.items():
                self._close_position(code, date, price)

            # 检查持仓数上限
            max_new = max(0, MAX_CONCURRENT - len(self.positions))
            eligible = [(c, p) for c, p in buy_today.items() if c not in self.positions]
            for code, price in eligible[:max_new]:
                self._open_position(code, date, price)

            self.equity_curve.append({
                'date': date,
                'cash': self.cash,
                'positions_value': sum(p.market_value for p in self.positions.values()),
                'total': self.cash + sum(p.market_value for p in self.positions.values()),
                'positions': {c: p.shares for c, p in self.positions.items()},
            })

        return self._compute_results()

    def _compute_results(self) -> dict:
        if len(self.equity_curve) < 2:
            return {'error': '数据不足'}

        df = pd.DataFrame(self.equity_curve)
        final_value = df['total'].iloc[-1]
        total_return = (final_value / self.initial_cash - 1) * 100

        # 日收益率
        df['daily_return'] = df['total'].pct_change()

        # 年化收益率
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        years = max(days / 365.0, 0.01)
        annual_return = ((final_value / self.initial_cash) ** (1 / years) - 1) * 100

        # 夏普比率
        risk_free = 0.02  # 2% 无风险利率
        excess = df['daily_return'] - risk_free / 252
        sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

        # 最大回撤
        df['cummax'] = df['total'].cummax()
        df['drawdown'] = (df['total'] - df['cummax']) / df['cummax'] * 100
        max_drawdown = df['drawdown'].min()

        # 交易统计
        trade_df = pd.DataFrame([{
            'code': t.code, 'date': t.date, 'direction': t.direction,
            'price': t.price, 'shares': t.shares, 'amount': t.amount,
        } for t in self.trades])

        win_trades = []
        buys = trade_df[trade_df['direction'] == 'buy']
        sells = trade_df[trade_df['direction'] == 'sell']
        paired = min(len(buys), len(sells))

        if paired > 0:
            for i in range(paired):
                b = buys.iloc[i]
                s = sells.iloc[i]
                pnl_pct = (s['price'] - b['price']) / b['price'] * 100
                win_trades.append(pnl_pct)
        else:
            # 持仓未平，按最新价估算
            for pos in self.positions.values():
                win_trades.append(pos.pnl_pct * 100)

        win_rate = (sum(1 for p in win_trades if p > 0) / len(win_trades) * 100) if win_trades else 0
        avg_win = np.mean([p for p in win_trades if p > 0]) if any(p > 0 for p in win_trades) else 0
        avg_loss = np.mean([p for p in win_trades if p <= 0]) if any(p <= 0 for p in win_trades) else 0

        return {
            'strategy': self.strategy.name,
            'start_date': str(df['date'].iloc[0]),
            'end_date': str(df['date'].iloc[-1]),
            'initial_cash': self.initial_cash,
            'final_value': round(final_value, 2),
            'total_return_pct': round(total_return, 2),
            'annual_return_pct': round(annual_return, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'total_trades': len(self.trades),
            'win_rate_pct': round(win_rate, 2),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'final_positions': len(self.positions),
            'equity_curve': df,
            'trades': self.trades,
        }

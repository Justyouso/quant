"""量化选股 - 全局配置"""

# 回测默认参数
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2025-12-31"
INITIAL_CASH = 1_000_000

# 技术指标参数
MA_PERIODS = [5, 10, 20, 60, 120, 250]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
KDJ_K = 9
KDJ_D = 3
BOLL_PERIOD = 20
BOLL_STD = 2

# 风控
MAX_POSITION_PCT = 0.25       # 单只股票仓位上限
MIN_VOLUME_RATIO = 0.01       # 最低换手率
MAX_CONCURRENT = 5             # 最大同时持仓数
TRADE_SLIPPAGE = 0.001         # 滑点 0.1%
COMMISSION_RATE = 0.00025     # 佣金万2.5
STAMP_TAX_RATE = 0.001        # 印花税千1（卖出）
MIN_COMMISSION = 5.0          # 最低佣金

# 选股
TOP_N = 10                    # 默认输出前N只
MIN_DAYS = 60                 # 计算指标最少需要的数据天数

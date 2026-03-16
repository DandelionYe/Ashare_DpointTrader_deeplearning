# constants.py
"""
全局常量集中管理。
所有模块从此处 import，避免跨模块依赖业务逻辑文件。
"""

# ==============================================================
# Walk-forward 训练约束 / 惩罚项
# （trainer_optimizer.py 和 reporter.py 共同使用）
# ==============================================================

# 硬约束：每折至少需要的已平仓交易数
MIN_CLOSED_TRADES_PER_FOLD: int = 2

# 软目标：惩罚项对齐的目标已平仓交易数
TARGET_CLOSED_TRADES_PER_FOLD: int = 4

# 惩罚项强度系数
LAMBDA_TRADE_PENALTY: float = 0.03

# ==============================================================
# 持久化文件名
# （trainer_optimizer.py 使用）
# ==============================================================
BEST_SO_FAR_FILENAME: str = "best_so_far.json"
BEST_POOL_FILENAME: str = "best_pool.json"
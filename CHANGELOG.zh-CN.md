# 更新日志

## [Ver3.0] - 2026-03-18

### 🚀 架构重构

* 对项目整体结构进行重构，提高模块化程度和可维护性
* 核心模块重新组织：

  * `training.py` 替代 `trainer_optimizer.py`
  * `evaluation.py` 替代 `backtester_engine.py` 和部分 `metrics.py`
  * `utils.py` 统一承载 run manifest 相关功能
* 移除旧模块，逻辑更加清晰

### 🧠 核心逻辑优化

* 修复交易可执行性判断逻辑：

  * 正确使用成交量（volume）进行流动性过滤
  * 修复 ST 开关和上市天数判断逻辑
* 提升回测稳定性：

  * 对缺失字段（如 `amount`）进行容错处理
  * 数据不完整时提供默认值

### 📊 指标与评估

* 重写 `trade_penalty` 逻辑：

  * 在目标交易次数处惩罚为 0
  * 偏离越大，惩罚越大
* 统一评估逻辑与测试预期

### 🧪 测试与 CI

* 所有测试适配新模块结构
* 移除旧模块依赖（如 `trainer_optimizer`、`backtester_engine`、`run_manifest`）
* 修复 CI 问题：

  * 移除无效依赖（如 `types-pandas`）
  * 修复 Python 版本兼容性
  * CI 环境中跳过 conda 重启

### ⚙️ CLI 改进

* 修复 `main_cli.py` 在 import 时执行副作用问题
* CLI 仅在 `__main__` 下运行
* 提升 CI 与测试环境兼容性

### 🧹 清理与简化

* 删除废弃文件：

  * `trainer_optimizer.py`
  * `backtester_engine.py`
  * `run_manifest.py`
* 项目结构更加简洁清晰

---

## [Ver2.0] - 上一版本

* 初步构建回测与训练框架
* 引入 CI、测试体系与模块化结构
* 实现基础的交易约束与评估逻辑

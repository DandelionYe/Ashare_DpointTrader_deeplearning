## [Ver3.5] - 2026-03-21

### 结构整理与模块统一
- 统一核心模块命名，提升代码结构清晰度与维护性：
  - `data.py` → `data_loader.py`
  - `training.py` → `trainer.py`
  - `evaluation.py` → `backtester.py`
  - `reporting.py` → `reporter.py`
- 同步更新主流程、测试与文档中的模块引用，使仓库结构与 README 描述保持一致。

### CLI 与 conda 环境处理改进
- 调整 conda 环境处理逻辑：
  - 默认模式下不再自动重启到 conda 环境；
  - 仅在显式传入 `--use-conda-env` 时才尝试切换；
  - 新增 `--target-conda-env`，用于提示预期环境名称。
- 改进环境切换逻辑的安全性与可控性，减少 IDE / CI / 调度环境中的误重启问题。

### 测试覆盖扩展
- 在原有测试基础上，新增以下覆盖：
  - 市场状态相关回测测试
  - conda 环境处理测试
  - optional torch runtime 测试
  - trainer calibration 测试
  - trainer split mode 测试
- 进一步增强了训练、回测、运行环境与兼容性相关的回归测试覆盖。

### 依赖与 CI 调整
- 新增 `requirements-dev.txt`，区分运行时依赖与开发/测试依赖。
- CI 安装流程改为优先使用 `requirements-dev.txt`，提高本地开发与自动化测试环境的一致性。

### 文档同步
- 更新 README 与中文 README，使其与当前仓库结构、模块命名和测试布局保持一致。
- 补充并整理版本变更说明，便于后续按 tag 追踪版本演进。
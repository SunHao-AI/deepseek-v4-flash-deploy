# Task 5 完成报告：engines/base.py + 注册表 + llamacpp 适配器

## 完成内容

- 创建 `script/engines/base.py`：
  - `RequirementError(RuntimeError)`：硬性条件不满足时拒绝启动。
  - `EngineAdapter(ABC)`：`__init__(profile, caps)` 存 `self.profile / self.caps / self.warnings`；抽象方法 `build_command`、`check_requirements`、`metrics_mapping`；可覆写方法 `health_url`（默认 `http://127.0.0.1:{port}/health`）、`pre_start`、`post_start`、`stop_patterns`（默认 []）、`api_key_args`（有 api_key 时返回 `["--api-key", key]`）。
- 创建 `script/engines/llamacpp.py`：`LlamaCppAdapter`，模块常量 `OFFICIAL_URL`、`DSPARK_PATTERNS`、`CTX_PER_SLOT`。
  - `check_requirements`：GPU 探测（gpu_count==0 → 抛错）、gpu_count 配置校验、model 存在性（无 download 段时）、DSpark 草稿发现与显存降级（<11GB 关闭）、显存预检（模型大小 ×1.1）。
  - `build_command`：参数与原脚本 `start_v4_flash_gguf.py` 一致。
  - `pre_start`：git clone / cmake 编译 / modelscope 下载。
  - `metrics_mapping`：四组指标映射。
  - `stop_patterns`：`["llama-server"]`。
  - 从原脚本原样搬运辅助函数：`run`、`require`、`find_server`、`_find_first`、`download_gguf`。
- 填充 `script/engines/__init__.py` 注册表：`get_adapter(engine)`，注册 `llamacpp`，未知引擎抛 `ProfileError`。
- 创建 `tests/test_engines_llamacpp.py`，含计划中 5 个测试用例。

## 关键实现点

- `find_server` 在找不到二进制时返回预期路径 `source/build/bin/llama-server`（而非 SystemExit），使测试无需真实编译产物；`pre_start` 会真正编译。
- 测试用 `probe(nvidia_smi_output=...)` 构造 8× 5880 Ada 场景，不调用真实 nvidia-smi。
- 代码注释用中文。
- 未修改 Task 1-4 已完成的文件（仅填充 `engines/__init__.py` 注册表）。

## 测试执行记录

1. 先运行测试确认失败：
   ```
   ImportError: cannot import name 'get_adapter' from 'engines'
   ```
2. 实现后运行 `tests/test_engines_llamacpp.py`：
   ```
   5 passed in 1.54s
   ```
3. 全量回归 `tests/`：
   ```
   44 passed in 5.81s
   ```

## Git 提交

```
feat(engines): 适配器基类与 llamacpp 适配器（迁移原启动脚本）
```

包含文件：
- `script/engines/base.py`
- `script/engines/llamacpp.py`
- `script/engines/__init__.py`
- `tests/test_engines_llamacpp.py`

# Task 6 完成报告：ollama 适配器

## 完成内容

- 创建 `script/engines/ollama.py`：`OllamaAdapter(EngineAdapter)`，额外公开方法 `unload_model()`。
  - `build_command`：`["ollama", "serve"]`；env 注入 `OLLAMA_HOST=0.0.0.0:{port}`、`OLLAMA_MODELS`（os.environ 有则透传）、`OLLAMA_NUM_PARALLEL`（默认 2）、有值时 `OLLAMA_CONTEXT_LENGTH`。
  - `check_requirements`：`caps.binaries.get("ollama")` 为 False → `RequirementError("ollama")`；model 缺失 → `RequirementError`。
  - `health_url`：`http://127.0.0.1:{port}/`（根路径）。
  - `pre_start`：`ollama list` 不含模型名 → `ollama pull <model>`。
  - `post_start`：`POST /api/generate` body `{"model", "keep_alive"}` 预加载（urllib，timeout 600）。
  - `unload_model`：`POST /api/generate` body `{"model", "keep_alive": 0}`，异常静默。
  - `metrics_mapping`：None（无 Prometheus 指标）。
  - `stop_patterns`：[]（serve 为共享常驻服务，不由 patterns 杀）。
- 修改 `script/engines/__init__.py`：import `OllamaAdapter` 并注册 `"ollama": OllamaAdapter`。
- 创建 `tests/test_engines_ollama.py`，含计划中 4 个测试用例。

## 关键实现点

- 测试用 `Capabilities(binaries={"ollama": True/False})` 控制二进制可用性，不调用真实 ollama。
- 代码注释用中文。
- 未修改 Task 1-5 已完成的文件（仅填充 `engines/__init__.py` 注册表）。

## 测试执行记录

1. 先运行测试确认失败：
   ```
   core.profile.ProfileError: 引擎未实现：ollama（已实现：['llamacpp']）
   ```
2. 实现后运行 `tests/test_engines_ollama.py`：
   ```
   4 passed in 0.31s
   ```
3. 全量回归 `tests/`：
   ```
   48 passed in 8.10s
   ```

## Git 提交

```
feat(engines): ollama 适配器（serve 共享、模型预加载/卸载）
```

包含文件：
- `script/engines/ollama.py`
- `script/engines/__init__.py`
- `tests/test_engines_ollama.py`

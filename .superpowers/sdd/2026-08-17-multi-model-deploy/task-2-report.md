# Task 2 报告：core/profile.py — YAML 加载、插值、校验

## 状态

完成 ✅

## 文件变更

| 文件 | 动作 | 说明 |
|---|---|---|
| `script/core/profile.py` | 新增 | Profile 数据类、ProfileError、KNOWN_ENGINES、load_profile、list_profiles、递归 ${VAR} 插值 |
| `tests/test_profile.py` | 新增 | 7 个测试用例 |

## 测试摘要

- **先运行确认失败**：`python -m pytest tests/test_profile.py -v` → `ModuleNotFoundError: No module named 'core.profile'`（符合预期）
- **实现后运行**：`python -m pytest tests/test_profile.py -v` → 7 passed
- **全量回归**：`python -m pytest tests/test_profile.py tests/test_envfile.py -v` → 10 passed

### 用例覆盖

1. `test_load_ok`：正常加载与插值
2. `test_missing_required_field`：缺少 port 时抛 ProfileError
3. `test_unknown_engine`：未知引擎 tensorrt 时抛 ProfileError
4. `test_interpolate_missing_var`：缺失环境变量时抛 ProfileError
5. `test_nested_interpolation`：嵌套映射（ollama.model）中的 ${ROOT} 插值
6. `test_list_profiles_sorted`：按文件名排序加载多个 profile
7. `test_missing_file`：文件不存在时抛 ProfileError

## Git 提交

- **哈希**：`e43617aae1c564dfc1377f83dc004eafb4914700`
- **消息**：`feat(core): profile YAML 加载、插值与校验`

## 注意事项

- 插值仅处理 `${VAR}` 格式，变量名符合 `[A-Za-z_][A-Za-z0-9_]*`。
- 变量未定义或值为空字符串均会抛出 `ProfileError`。
- `engine_config` 取与 `engine` 同名的 YAML 段；缺失该段时默认空字典。
- `KNOWN_ENGINES` 目前仅包含计划约定的四个引擎：`llamacpp`、`ollama`、`vllm`、`sglang`。
- `load_profile` 中 `models_dir` 默认指向 `PROJECT_ROOT / "models"`，与 `list_profiles` 保持一致。
- 未修改 Task 1 已完成的 `script/core/envfile.py`。

# models 目录重构与模型自动下载设计

## 背景

当前 `models/` 目录下所有 profile 都平铺在根目录，文件名需要加引擎后缀（如 `qwen3-llama.yaml`）。随着引擎和模型增多，这种方式不够清晰。同时，llama.cpp 需要用户手动下载 GGUF 模型并填写本地路径，体验不佳。

## 目标

1. 按引擎将 profile 分目录存放，文件名不再需要引擎后缀。
2. 保持对旧 `models/*.yaml` 的向后兼容。
3. 各引擎支持通过 ModelScope 自动下载模型；下载完成后将本地路径持久化写回 YAML。
4. 为所有 profile YAML 增加详细注释，说明各字段含义。
5. 更新 README.md 和后台启动指南。

## 设计

### 1. 目录结构

```text
models/
├── deepseek-v4.yaml          # 旧根目录文件，继续兼容
├── qwen3-ollama.yaml
├── qwen3-vllm.yaml
├── qwen3-llama.yaml
├── llamacpp/
│   └── qwen3.yaml            # 新位置，name 无需引擎后缀
├── ollama/
│   └── qwen3.yaml
├── vllm/
│   └── qwen3.yaml
└── sglang/
    └── (future).yaml
```

### 2. Profile 加载规则

- `load_profile(name)` 优先查找 `models/<name>.yaml`；不存在时递归查找 `models/*/<name>.yaml`。
- `list_profiles()` 递归扫描 `models/` 下所有 `.yaml`。
- 若同一个 `name` 同时出现在根目录和子目录，以根目录为准（兼容优先），并在日志中提示。

### 3. 模型自动下载与持久化

#### 3.1 公共下载接口

在 `src/modelctl/engines/` 中新增一个轻量下载工具函数，统一处理 ModelScope 下载：

```python
def download_from_modelscope(modelscope_id: str, local_root: Path) -> Path:
    """下载 ModelScope 模型仓库到本地，返回本地目录。"""
```

各引擎按需调用：

- **llamacpp**：`download_gguf()` 保持原有逻辑，下载指定 `quant` 的 GGUF 分片，返回首个 `.gguf` 文件路径。
- **vllm / sglang**：调用 `download_from_modelscope()` 下载整个 HF 格式仓库到 `MODEL_ROOT/<modelscope_id>`，返回本地目录路径。
- **ollama**：继续由 `ollama pull` 自行管理，不新增 download 段。

#### 3.2 下载触发条件

在引擎适配器的 `pre_start()` 中：

1. 读取 `cfg["model"]`。
2. 若 `model` 指向的本地文件（llamacpp）或目录（vllm/sglang）已存在，直接使用，不写回。
3. 若不存在且配置了 `download`，执行下载。
4. 下载成功后，将 `model` 更新为下载得到的本地路径，并写回原 YAML 文件。

#### 3.3 持久化写回 YAML

- 写回时仅修改对应引擎段下的 `model` 字段，保留其他字段。
- 写回前对原文件做备份：`qwen3.yaml.bak`。
- 使用 `PyYAML` 的 `safe_dump`，注释会丢失是已知限制；为弥补，建议保留字段顺序和缩进。
- 写回后继续后续启动流程，不重新加载 profile。

### 4. YAML 注释规范

所有 profile 增加统一注释：

- 顶层字段含义：`name`、`engine`、`port`、`api_key`。
- 引擎字段：每个字段的作用、默认值、影响。
- `download` 段：如何触发自动下载、下载后 `model` 如何被覆盖。
- 量化/模型选择建议：根据显存给出推荐。

示例 `models/llamacpp/qwen3.yaml`：

```yaml
# qwen3.yaml —— Qwen3.8-27B GGUF，llama.cpp 引擎
name: qwen3
engine: llamacpp
port: 8000
api_key: ${API_KEY}

llamacpp:
  # 本地 GGUF 文件路径。若为空或文件不存在，则通过 download 段自动下载；
  # 下载成功后会被覆盖为本地的绝对路径。
  model: ""

  # ModelScope 自动下载配置
  download:
    modelscope_id: unsloth/Qwen3.8-27B-GGUF
    quant: Q4_K_M          # 仓库内量化名，如 Q4_K_M / Q5_K_M / Q8_0 / UD-Q8_K_XL

  parallel: 2              # 并发槽位数
  ctx_size: 32768          # 单槽上下文长度；留空 = parallel × 1M
  reasoning: on            # 输出 reasoning 内容
  reasoning_format: deepseek
  dspark: off              # Qwen 无 DSpark，必须关闭
  cache_type_k: q8_0       # K cache 量化
  cache_type_v: q8_0       # V cache 量化
  gpu_count: 2             # 实际 GPU 数，决定 --tensor-split
  fit: off                 # llama.cpp 显存适配策略

usage:
  price_in: 0.5
  price_out: 1.0
```

### 5. 文档更新

#### README.md

- 更新目录结构说明，指出 `models/<engine>/*.yaml` 是新推荐方式，同时兼容旧的 `models/*.yaml`。
- 更新启动示例，使用新路径。
- 说明 `download` 段与 model 自动下载/写回机制。

#### docs/DeepSeek-V4-Flash后台启动指南.md

- 在"目录布局"和"配置说明"章节补充子目录规则。
- 补充 model 自动下载与持久化说明。

## 依赖

- 已依赖 `PyYAML`。
- 已依赖 `modelscope`（llamacpp 下载处会按需安装）。

## 风险与注意事项

1. **YAML 注释丢失**：使用 `PyYAML` 写回会丢弃注释，建议通过详细的字段命名和文档来弥补。
2. **同名 profile 歧义**：根目录与子目录同名时以根目录优先，需在日志中明确提示。
3. **下载失败回退**：下载失败时应保留原 `model` 字段不变，避免破坏配置文件。
4. **vllm/sglang 模型格式**：ModelScope 上的 HF 格式仓库与 GGUF 仓库不同，profile 中 `modelscope_id` 必须对应引擎可识别的格式。

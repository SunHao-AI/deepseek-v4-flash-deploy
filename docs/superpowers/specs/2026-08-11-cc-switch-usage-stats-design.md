# cc-switch 用量统计服务设计

- 日期：2026-08-11
- 状态：已批准（待实现）

## 1. 背景与目标

本地通过 llama.cpp `llama-server`（DeepSeek-V4-Flash-0731 GGUF，节点 210，端口 18888）提供 OpenAI 兼容接口。希望在 **cc-switch** 的供应商卡片上实时展示：

- 自服务启动以来累计消耗的 token 数（输入 / 输出分别展示）
- 当前生成速率（tokens/s）
- 按 DeepSeek-V4-Flash 官方 API 价格折算的累计费用（人民币）

llama-server 本身没有 `/api/usage` 这类用量查询端点，需要通过一个轻量统计服务聚合其 `/metrics`（Prometheus）数据并伪装成 cc-switch 可识别的接口。

## 2. 现状架构（实测）

```
cc-switch ──> nginx（监听 :5000，sites-enabled/myflaskapp）
               ├─ location ~ ^/210/llm/(.*)$ ──> proxy_pass http://192.168.77.210:18888/$1   （LLM API 主流量）
               └─ location / ──> 127.0.0.1:5001（已占用，统计服务不得使用 5001）
```

- cc-switch 供应商卡片的 API 请求地址（baseUrl）形如 `http://<nginx>:5000/210/llm/v1`
- cc-switch 用量查询请求 URL = `{{baseUrl}}/api/usage`，即 `http://<nginx>:5000/210/llm/v1/api/usage`
- 该路径目前会被 `^/210/llm/(.*)$` 命中并转发到 llama-server（18888），返回 404

## 3. 目标架构

```
┌────────────┐  GET /210/llm/v1/api/usage
│ cc-switch  │──────────────────────────┐
└────────────┘                          ▼
                               ┌──────────────────┐
                               │ nginx :5000      │
                               │ 新增精确匹配      │
                               │ location（位于    │
                               │ ^/210/llm/(.*)$  │
                               │ 之前）           │
                               └──────────────────┘
                                       │
                                       ▼
                            ┌────────────────────┐
                            │ 统计服务            │
                            │ 127.0.0.1:5002     │
                            │ /api/usage         │
                            └────────────────────┘
                                       │ 每 5s 轮询
                                       ▼
                            ┌────────────────────┐
                            │ llama-server       │
                            │ 192.168.77.210:    │
                            │ 18888/metrics      │
                            │（需 --metrics）     │
                            └────────────────────┘
```

原则：**API 主流量路径完全不变**；统计服务只新增一条 nginx 精确匹配路由，零侵入 llama-server 与网关。

## 4. 组件设计

### 4.1 llama-server 改动

- 启动命令追加 `--metrics`（启用 `/metrics` Prometheus 端点），对推理无影响
- 涉及文件：
  - `script/start_v4_flash_gguf.py`：启动命令列表追加 `--metrics`
  - `script/start_v4_flash_background.sh`：`ARGS` 追加 `--metrics`
- 生效方式：重启服务

### 4.2 统计服务（新增 `script/usage_stats_server.py`）

**技术选型**：Python 标准库单文件实现（`http.server` + `urllib` + 内置 `.env` 解析），零第三方依赖，与项目「运行期零第三方依赖」风格一致。

**功能**：

1. **定时轮询**（默认每 5 秒）llama-server `/metrics`
2. **指标解析**（容错匹配新旧版本指标名，正则匹配）：

   | 用途 | 旧版指标名 | 新版指标名（`llamacpp:` 前缀） |
   |---|---|---|
   | 累计输入 tokens | `prompt_tokens_total` | `llamacpp:tokens_evaluated_total` |
   | 累计输出 tokens | `tokens_predicted_total` | `llamacpp:tokens_predicted_total` |
   | 生成速率 tok/s | `predicted_tokens_seconds`（gauge） | `llamacpp:tokens_predicted_seconds` |

3. **费用计算**（.env 可配置，官方价默认）：

   ```
   费用 = 累计输入tokens / 1e6 × PRICE_IN + 累计输出tokens / 1e6 × PRICE_OUT
   PRICE_IN  = 1.0 元/M（缓存未命中价；本地无缓存命中概念）
   PRICE_OUT = 2.0 元/M
   ```

4. **`/api/usage` 响应**（cc-switch extractor 直接消费）：

   ```json
   {
     "isValid": true,
     "used": 12.34,
     "total": 100.0,
     "remaining": 87.66,
     "unit": "CNY",
     "planName": "DeepSeek-V4-Flash 本地部署",
     "extra": "累计 1,234,567 tokens（输入 823,456 / 输出 411,111）| 生成速率 55.2 tok/s"
   }
   ```

   - `remaining = total - used`；`total`（预算）由 `.env` 的 `USAGE_BUDGET` 配置，不配置时 `total/remaining` 为 null
   - 速率展示在 `extra`，取值优先级：(1) 若 `/metrics` 中存在 gauge 类型速率指标（如旧版 `predicted_tokens_seconds`）直接取；(2) 否则用轮询差值法：两次轮询的累计输出 tokens 差值 ÷ 时间间隔

### 4.3 nginx 配置（新增）

放在 `location ~ ^/210/llm/(.*)$` **之前**（nginx 正则 location 按顺序匹配、首个命中生效）：

```nginx
# ---- 210 LLM 用量统计（cc-switch 用量查询）----
location ~ ^/210/llm/v1/api/usage$ {
    proxy_pass http://127.0.0.1:5002/api/usage;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

### 4.4 cc-switch 配置（粘贴到「用量查询 → 自定义」）

```js
({
  request: {
    url: "{{baseUrl}}/api/usage",
    method: "GET",
    headers: { "Authorization": "Bearer {{apiKey}}", "User-Agent": "cc-switch/1.0" }
  },
  extractor: function(response) {
    if (!response || response.error || response.isValid === false) {
      return { isValid: false, invalidMessage: (response && (response.invalidMessage || (response.error && response.error.message))) || "接口调用失败" };
    }
    return {
      isValid: true,
      used: response.used,
      remaining: response.remaining,
      total: response.total,
      unit: response.unit || "CNY",
      planName: response.planName,
      extra: response.extra
    };
  }
})
```

### 4.5 配置项（.env.example 新增段）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `USAGE_PORT` | `5002` | 统计服务监听端口 |
| `USAGE_LLAMA_BASE` | `http://192.168.77.210:18888` | llama-server 地址 |
| `USAGE_POLL_INTERVAL` | `5` | 轮询间隔（秒） |
| `USAGE_PRICE_IN` | `1.0` | 输入单价（元/M tokens） |
| `USAGE_PRICE_OUT` | `2.0` | 输出单价（元/M tokens） |
| `USAGE_BUDGET` | 空 | 预算（元），空则不显示 total/remaining |
| `LLAMA_API_KEY` | 空 | 若 llama-server `/metrics` 要求鉴权时透传 Bearer |

## 5. 数据流

1. cc-switch 发起 `GET {{baseUrl}}/api/usage`
2. nginx 精确匹配 `/210/llm/v1/api/usage` → 转发到 `127.0.0.1:5002/api/usage`
3. 统计服务返回最近一次轮询结果（内存态，含缓存时间戳）
4. cc-switch extractor 提取字段并展示到供应商卡片

## 6. 错误处理与边界

- llama-server 未启动 / `/metrics` 未启用 → 统计服务返回 `{"isValid": false, "invalidMessage": "..."}`，cc-switch 卡片显示红色失效提示
- 首次轮询完成前 → 返回全 0 数据（不误导）
- `/metrics` 若要求鉴权 → 统计服务透传 `LLAMA_API_KEY` 作为 Bearer
- 统计服务为单文件常驻进程，崩溃时手动重启（不做守护化，保持简单，YAGNI）
- 指标名容错：新旧版本前缀均支持；都无法匹配时视为服务不可用

## 7. 改动文件清单

| 文件 | 动作 |
|---|---|
| `script/start_v4_flash_gguf.py` | 修改：追加 `--metrics` |
| `script/start_v4_flash_background.sh` | 修改：ARGS 追加 `--metrics` |
| `script/usage_stats_server.py` | 新增：统计服务 |
| `.env.example` | 修改：新增用量统计配置段 |
| nginx `sites-enabled/myflaskapp` | 修改（服务器侧）：新增精确匹配 location |
| cc-switch | 配置（用户侧）：粘贴用量查询配置 |

## 8. 非目标

- 不做按时间段的用量统计（如今日/本月），仅自服务启动以来累计
- 不做多节点（209 等）统计，仅 210
- 不做统计服务守护化、不做历史持久化
- 不做费用汇率换算（仅 CNY）

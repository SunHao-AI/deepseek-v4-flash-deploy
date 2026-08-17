# Task 8 报告：core/stats.py — 用量统计多引擎适配

## 状态
完成。

## 交付物
- 新建 `script/core/stats.py`（迁移自 `script/usage_stats_server.py`）
- 新建 `tests/test_stats.py`（4 个 pytest 用例）

## 实现要点
- `parse_metrics(text, mapping)`：保留现版预编译正则 `^name(?:\{[^}]*\})?\s+([0-9.eE+-]+)$`，仅把全局 `METRIC_NAMES` 改为参数 `mapping`；按各键候选名取第一个命中，无命中 0.0。
- `build_usage_payload(tokens, usage_cfg, start_time, now)`：按 `price_in/price_out/budget` 折算。**输出字段与现版完全一致**：`isValid/used/unit/planName/extra/total/remaining`。无预算时 `total`/`remaining` 为 `None`（现版语义，字段仍存在）。`start_time/now` 用于在 `extra` 中追加运行时长。
- `@dataclass StatsTarget`：`name/metrics_url/mapping/usage_cfg/api_key`。
- `run_server(targets)`：`ThreadingHTTPServer`，`/api/usage` 与 `/api/usage?model=<name>`；保留 poll/on-demand 两种模式与 `USAGE_HOST/USAGE_PORT/USAGE_MODE/USAGE_POLL_INTERVAL` 环境变量；`mapping` 为 None（ollama）返回 `{"error": "该引擎不支持精确统计"}` HTTP 200；`?model` 缺省返回 targets 第一个。
- `UsageCollector` 保留 poll/on-demand 逻辑与速率回退计算（`predicted_rate<=0` 时用 delta 计算），仅新增 `mapping` 参数。
- 提供 `if __name__ == "__main__"` 入口 + `main()`，支持 `python -m core.stats` 独立运行（供 Task 9 后台化），从 `models/*.yaml` 构造 targets。

## 测试
- `tests/test_stats.py` 4 用例：`parse_metrics_llamacpp`、`parse_metrics_vllm_no_rate`、`build_payload_with_budget`、`build_payload_no_budget`。
- **字段名调整**：计划原测试用 `used_cost`，现版实际字段为 `used`，已按现版改为 `payload["used"]`；无预算用例由 `"total" not in payload` 改为 `payload["total"] is None`（现版语义为字段存在但为 None）。
- 全量回归：`python -m pytest tests/ -v` → **57 passed**（含旧 `test_usage_stats_server.py` 20 用例，仍针对旧文件，Task 10 删除）。

## 提交
- commit `99a2c37`：`feat(core): 用量统计迁移至多引擎指标映射（/api/usage 兼容）`
- 仅提交本任务两个新文件；`script/usage_stats_server.py` 的既有修改未触碰（Task 10 处理删除）。

## 备注
- 未修改 Task 1-7 文件；未删除旧文件。

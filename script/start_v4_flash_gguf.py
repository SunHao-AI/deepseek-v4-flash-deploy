#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================
# @File   : script/start_v4_flash_gguf.py
# @IDE    : VSCode
# @Author : Unknown
# @Email  :
# @Date   : 2026/8/7 14:58
# @Desc   : 启动 DeepSeek-V4-Flash GGUF 推理服务
# ===============================================================================

"""Build an official llama.cpp (CUDA) and serve DeepSeek-V4-Flash-0731 GGUF.

Switched from the Fringe210 fork to the OFFICIAL llama.cpp (ggml-org/llama.cpp),
which now natively supports the DeepSeek-V4 architecture (PR #22607 / #24162)
and DSpark speculative decoding (PR #25784, with multi-GPU improvements).

Why GGUF and not the official FP4/FP8 safetensors checkpoint:
  The 8x RTX 5880 Ada (CC 8.9) cannot natively serve DeepSeek's FP4/FP8
  checkpoint in stock vLLM (native FP4 needs Blackwell SM100/SM120). The
  llama.cpp GGUF route is the supported path on this hardware.

DSpark:
  DSpark is enabled for DeepSeek-V4-Flash-0731 GGUF, giving ~1.5x-1.9x faster
  decoding. It needs an extra drafter GGUF (`dspark-DeepSeek-V4-Flash-0731-Q8_0`)
  loaded via --model-draft. It costs roughly +10 GB VRAM.

  Official Unsloth recommendation: `--spec-draft-n-max 3` (larger values slower).

KV cache quantization (--cache-type-k/--cache-type-v):
  Defaults to q8_0/q8_0. CONFIRMED WORKING on official llama.cpp b10298 on this
  machine (8x RTX 5880 Ada): normal output, ~55 tok/s single stream, DSpark
  acceptance ~50%. The q8_0/q4_0 degenerate-output bug reported earlier was
  CONFIRMED on the Fringe210 fork for Gated Delta Net; it does NOT reproduce on
  this official build. f16 remains the conservative fallback via
  --cache-type-k f16 --cache-type-v f16, or pass --cache-type-k "" to disable.

--parallel:
  Defaults to 2. The GGML_ASSERT crash in llm_build_deepseek4 with --parallel 2
  was reported on the Fringe210 fork (Gated Delta Net multi-sequence recurrent
  state reshape); official llama.cpp b10298 was NOT previously verified at 2,
  so confirm normal output after switching. Use --parallel 1 if it regresses.

--fit:
  Defaults to "off". With --fit on llama.cpp cannot measure the draft model
  memory in a standalone context, logs "[spec] failed to measure draft model
  memory" and skips the reserve -- the ~11GB drafter then falls outside the
  fitting budget, risking OOM on tight VRAM. --fit off pins placement and is
  what the DSpark README recommends.

--no-mmap:
  Removed from the default launch command. mmap (on by default in llama.cpp)
  allows demand paging and usually faster loading; the flag was carried over
  from the fork but is not needed on the official build with fully-resident
  weights.

--quant:
  Defaults to UD-Q8_K_XL (the lossless 8-bit build, bit-identical experts and
  non-expert BF16 -> no rounding). UD-Q4_K_XL is near-lossless (experts
  bit-exact, non-experts to Q8_0) and ~7GB smaller; both fit comfortably on
  8x RTX 5880 Ada (384GB).

Example:
  python3 script/start_v4_flash_gguf.py \
    --model /raid5/sh/model-gguf/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


OFFICIAL_URL = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_MODELSCOPE_ID = "unsloth/DeepSeek-V4-Flash-0731-GGUF"
# The DSpark drafter lives in the same repo, in the repo root (not under <quant>/).
DSPARK_PATTERNS = ["*dspark*"]
# 每个并发槽位的默认上下文长度（DeepSeek-V4-Flash-0731 支持的最大 1M 上下文）。
CTX_PER_SLOT = 1_048_576
# 脚本所在目录，作为所有默认相对路径的基准。
# 默认目录布局（脚本位于项目根的 script/ 子目录）：
#   <项目根>/script/        e.g. /raid5/sh/code/deepseek-v4-flash/script/
#   <项目根>/../llama.cpp   e.g. /raid5/sh/code/llama.cpp
#   <项目根>/../../model-gguf  e.g. /raid5/sh/model-gguf
#   <项目根>/../../logs     e.g. /raid5/sh/logs
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# 配置加载：优先级 命令行参数 > 环境变量 > .env > 脚本内置默认值
# ---------------------------------------------------------------------------
def env(key: str, default: str | None = None) -> str | None:
    """读取环境变量（含 .env 注入），未设置时返回 default。"""
    return os.environ.get(key, default)


def _parse_env_file(path: Path) -> dict[str, str]:
    """轻量解析 .env：KEY=VALUE，忽略空行与 # 注释，去除首尾引号。"""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def load_env(env_path: Path | None = None) -> Path:
    """把 .env 载入 os.environ（已存在的环境变量优先，不覆盖）。

    优先使用 python-dotenv（若已安装），否则回退到内置解析器，
    因此本脚本不强制依赖任何第三方库。
    返回实际检查的 .env 路径（不存在时也返回该路径）。
    """
    path = env_path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return path
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
    except ImportError:
        for key, value in _parse_env_file(path).items():
            os.environ.setdefault(key, value)
    return path


def _env_bool(key: str, default: bool) -> bool:
    """把环境变量解析为布尔值（on/true/1/yes 视为 True）。"""
    raw = env(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("on", "true", "1", "yes")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def require(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"缺少 {name}。请安装后再运行脚本。")


def find_server(source: Path) -> Path:
    for candidate in (source / "build" / "bin" / "llama-server", source / "llama-server"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise SystemExit("编译结束但找不到 llama-server。请检查 CMake 输出。")


def _find_first(destination: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(destination.rglob(pattern))
        if matches:
            return matches[0]
    return None


def download_gguf(
    modelscope_id: str, model_root: Path, quant: str, want_dspark: bool
) -> tuple[Path, Path | None]:
    """Download only the given quant's shards (plus the DSpark drafter if wanted).

    Returns (model_first_shard, draft_path_or_None). Never downloads the whole repo.
    """
    if importlib.util.find_spec("modelscope") is None:
        run([sys.executable, "-m", "pip", "install", "-U", "modelscope"])

    from modelscope import snapshot_download

    destination = model_root / modelscope_id.rsplit("/", 1)[-1]
    destination.mkdir(parents=True, exist_ok=True)

    patterns = [f"*{quant}*-00001-of-*.gguf", f"*{quant}*.gguf"]
    if want_dspark:
        patterns.extend(DSPARK_PATTERNS)

    print(f"从 ModelScope 下载 {modelscope_id} 的 {quant} 分片（{', '.join(patterns)}）：{destination}")
    try:
        snapshot_download(
            model_id=modelscope_id,
            local_dir=str(destination),
            allow_file_pattern=patterns,
        )
    except Exception as error:
        raise SystemExit(
            f"ModelScope 下载失败。该 0731 GGUF 镜像可能尚未同步 {quant} 分片，或需要登录。\n"
            f"模型 ID：{modelscope_id}\n"
            "可尝试传入 --modelscope-id 指定实际镜像 ID，或从可访问 Hugging Face 的机器下载后传入 --model。"
        ) from error

    # ModelScope preserves the repository's <quant>/ subdirectory, so the
    # shards are not necessarily directly under ``destination``.
    model_match = _find_first(
        destination, [f"*{quant}*-00001-of-*.gguf", f"*{quant}*.gguf"]
    )
    if model_match is None:
        raise SystemExit(
            f"下载结束，但未找到 {quant} 的首分片：{destination}\n"
            "请检查 ModelScope 仓库中的实际量化名称；脚本没有下载官方 safetensors 权重。"
        )

    draft_match = None
    if want_dspark:
        draft_match = _find_first(destination, DSPARK_PATTERNS)
        if draft_match is None:
            print(
                f"\n警告：未在 {destination} 下找到 DSpark 草稿文件（{'、'.join(DSPARK_PATTERNS)}）。\n"
                "DSpark 将不会启用；请确认 ModelScope 镜像已同步 dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf，\n"
                "或改用 Hugging Face 手动下载后通过 --draft 指定。\n",
                flush=True,
            )
        else:
            print(f"找到 DSpark 草稿：{draft_match}")

    return model_match, draft_match


def main() -> None:
    env_file = load_env()
    if env_file.is_file():
        print(f"已加载配置：{env_file}", flush=True)

    parser = argparse.ArgumentParser(
        description="在 CUDA/Ada GPU 上启动 DeepSeek-V4-Flash GGUF 的 OpenAI 兼容服务（官方 llama.cpp + DSpark）。"
    )
    parser.add_argument("--model", type=Path, default=env("MODEL"),
                        help="V4-aware GGUF 的第一个分片，例如 ...-00001-of-00005.gguf（.env: MODEL）")
    parser.add_argument("--draft", type=Path, default=(env("DRAFT") or None),
                        help="DSpark 草稿 GGUF（dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf）。"
                             "不指定时脚本会自动在模型同目录及各级上级目录中查找 dspark*.gguf。")
    parser.add_argument("--download", action="store_true",
                        help="从 ModelScope 下载指定量化版本的 GGUF；只下载该量化版本的分片")
    parser.add_argument("--quant", default=env("QUANT", "UD-Q8_K_XL"),
                        help="要下载的量化版本目录名，例如 UD-Q8_K_XL（默认，无损）、UD-Q4_K_XL（近无损），"
                             "对应 ModelScope 仓库里的子目录名，仅在 --download 时生效")
    parser.add_argument("--modelscope-id", default=env("MODELSCOPE_ID", DEFAULT_MODELSCOPE_ID),
                        help="ModelScope GGUF 仓库 ID")
    parser.add_argument("--model-root", type=Path,
                        default=(Path(env("MODEL_ROOT")) if env("MODEL_ROOT")
                                 else PROJECT_ROOT.parent.parent / "model-gguf"),
                        help="--download 时 GGUF 的保存父目录（默认 <项目根>/../../model-gguf）")
    parser.add_argument("--source-dir", type=Path,
                        default=(Path(env("SOURCE_DIR")) if env("SOURCE_DIR")
                                 else PROJECT_ROOT.parent / "llama.cpp"),
                        help="官方 llama.cpp 源码目录（默认 <项目根>/../llama.cpp）")
    parser.add_argument("--port", type=int, default=int(env("PORT", "18888")))
    parser.add_argument("--api-key", type=str, default=(env("API_KEY") or None),
                        help="llama-server 的 API 密钥。设置后所有请求需带 "
                             "Authorization: Bearer <key>；不设置则不校验。")
    # --ctx-size 不设固定默认值：命令行显式传入优先，其次 .env 的 CTX_SIZE，
    # 均未指定时按 CTX_PER_SLOT × --parallel 自动计算（每个并发 1M 上下文）。
    parser.add_argument("--ctx-size", type=int, default=argparse.SUPPRESS,
                        help="总上下文长度。默认按并发数自动计算：PARALLEL × 1048576"
                             "（每个并发槽位 1M 上下文）；可用命令行或 .env 的 CTX_SIZE 覆盖。")
    parser.add_argument("--parallel", type=int, default=int(env("PARALLEL", "2")),
                        help="并发序列数，默认 2。若在官方版上出现并行崩溃，用 --parallel 1 回退。")
    parser.add_argument("--gpu-count", type=int, default=int(env("GPU_COUNT", "8")))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--reasoning", default=env("REASONING", "on"), choices=["on", "off"],
                        help="是否开启推理/思考模式，默认 on")
    parser.add_argument("--reasoning-format", default=env("REASONING_FORMAT", "deepseek"),
                        help="推理内容的输出格式，默认 deepseek")
    parser.add_argument("--fit", default=env("FIT", "off"),
                        help="传给 llama-server 的 --fit 参数，默认 off（DSpark README 推荐）。"
                             "--fit on 时 llama-server 无法为草稿模型单独测量显存、跳过预留，"
                             "草稿 ~11GB 会落在预算外，显存紧张时可能 OOM。传 on 开启。")
    parser.add_argument("--cache-type-k", default=env("CACHE_TYPE_K", "q8_0"),
                        help="KV cache K 量化类型，默认 q8_0。已在官方 b10298 上验证正常输出；"
                             "如需保守用 f16 传 --cache-type-k f16，禁用传空串 \"\"。")
    parser.add_argument("--cache-type-v", default=env("CACHE_TYPE_V", "q8_0"),
                        help="KV cache V 量化类型，默认 q8_0。同上。")
    parser.add_argument("--no-dspark", action="store_true", default=not _env_bool("DSPARK", True),
                        help="禁用 DSpark 投机解码（不加载草稿模型、不加 spec 参数）。"
                             "默认由 .env 的 DSPARK 控制（on=启用）。")
    parser.add_argument("--spec-type", default=env("SPEC_TYPE", "draft-dspark"),
                        help="投机解码类型，默认 draft-dspark")
    parser.add_argument("--spec-draft-n-max", type=int, default=int(env("SPEC_DRAFT_N_MAX", "3")),
                        help="DSpark 最大草稿 token 数，默认 3（Unsloth 推荐，更大反而更慢）")
    parser.add_argument("--n-gpu-layers-draft", type=int, default=int(env("N_GPU_LAYERS_DRAFT", "999")),
                        help="草稿模型 GPU 层数，默认 999")
    parser.add_argument("--log-dir", type=Path,
                        default=(Path(env("LOG_DIR")) if env("LOG_DIR")
                                 else PROJECT_ROOT.parent.parent / "logs"),
                        help="服务器日志文件的保存目录（默认 <项目根>/../../logs）")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="服务器日志文件的完整路径。若不指定，则在 --log-dir 下"
                             "自动生成 llama-server-<port>-<时间戳>.log")
    parser.add_argument("--no-console", action="store_true",
                        help="只写日志文件，不在终端打印输出（默认终端和文件同时输出）")
    parser.add_argument("--no-log-append", action="store_true",
                        help="覆盖已存在的日志文件，而不是追加写入")
    args = parser.parse_args()

    # --ctx-size 解析：命令行显式传入 > .env 的 CTX_SIZE > 按并发自动计算（每并发 1M）
    if not hasattr(args, "ctx_size"):
        env_ctx = env("CTX_SIZE")
        if env_ctx:
            args.ctx_size = int(env_ctx)
        else:
            args.ctx_size = args.parallel * CTX_PER_SLOT
            print(
                f"未指定 --ctx-size，按并发数自动计算：{args.parallel} × {CTX_PER_SLOT}"
                f" = {args.ctx_size}（每个并发槽位 1M 上下文）",
                flush=True,
            )

    if not 1 <= args.port <= 65535:
        raise SystemExit("端口必须在 1 到 65535 之间。")
    if args.ctx_size < 1024:
        raise SystemExit("--ctx-size 至少应为 1024。")
    if args.model is None and not args.download:
        raise SystemExit("请提供 --model，或加入 --download 自动从 ModelScope 下载（配合 --quant 指定量化版本）。")
    if args.model is not None and args.download:
        raise SystemExit("--model 与 --download 只能选一个。")

    # If the user overrides KV cache quants away from the verified defaults
    # (q8_0/q8_0), point out that q8_0 was the validated setting on this build.
    if (args.cache_type_k or args.cache_type_v) and (
        args.cache_type_k != "q8_0" or args.cache_type_v != "q8_0"
    ):
        print(
            "\n注意：你改动了 KV cache 量化类型 "
            f"(K={args.cache_type_k or 'f16'}, V={args.cache_type_v or 'f16'})。"
            "默认的 q8_0/q8_0 已在官方 b10298 上验证输出正常；"
            "q4_0 或 f16 也可用，但如需恢复到验证过的组合请去掉这两个参数。\n",
            flush=True,
        )

    want_dspark = not args.no_dspark
    draft_path: Path | None = args.draft

    if args.download:
        model_path, auto_draft = download_gguf(
            args.modelscope_id, args.model_root.expanduser().resolve(), args.quant, want_dspark
        )
        if draft_path is None:
            draft_path = auto_draft
    else:
        model_path = args.model.expanduser()

    if not model_path.is_file():
        raise SystemExit(
            f"找不到 GGUF 模型：{model_path}\n"
            "请先下载 V4 专用 GGUF（例如 Unsloth 的 0731 Q4_K_XL），"
            "并把 --model 指向 -00001-of-0000N.gguf。"
        )

    if want_dspark and draft_path is None and not args.download:
        # Auto-discover a dspark drafter: first next to the model, then in each
        # parent directory (ModelScope/HF preserve a <quant>/ subdir, so the
        # drafter often lives one level up, e.g. at the repo root).
        for base in (model_path.parent, *model_path.parents):
            if not base.exists():
                continue
            # Direct hit (preferred): the well-known filename right in this dir.
            for name in (
                "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
                "dspark-DeepSeek-V4-Flash-0731-BF16.gguf",
            ):
                candidate = base / name
                if candidate.is_file():
                    draft_path = candidate
                    break
            if draft_path is not None:
                break
            # Fallback: any *dspark*.gguf in this directory only.
            found = sorted(p for p in base.glob("*dspark*.gguf") if p.is_file())
            if found:
                draft_path = found[0]
                break

    if want_dspark and draft_path is not None:
        draft_path = draft_path.expanduser().resolve()
        if not draft_path.is_file():
            raise SystemExit(f"找不到 DSpark 草稿模型：{draft_path}")
        print(f"使用 DSpark 草稿模型：{draft_path}")
    else:
        print("未启用 DSpark（未找到草稿模型或 --no-dspark）。", flush=True)

    require("git")
    require("cmake")
    require("nvidia-smi")
    run(["nvidia-smi", "-L"])

    source = args.source_dir.expanduser().resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", OFFICIAL_URL, str(source)])
    elif not (source / ".git").is_dir():
        raise SystemExit(f"{source} 已存在但不是 Git 仓库；不会覆盖它。")

    if not args.skip_build:
        run([
            "cmake", "-S", str(source), "-B", str(source / "build"),
            "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
        ])
        run(["cmake", "--build", str(source / "build"), "--config", "Release", "-j"])

    if args.build_only:
        print("构建完成。", flush=True)
        return

    server = find_server(source)
    gpu_split = ",".join(["1"] * args.gpu_count)
    command = [
        str(server),
        "--model", str(model_path.resolve()),
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--ctx-size", str(args.ctx_size),
        "--parallel", str(args.parallel),
        "--n-gpu-layers", "999",
        "--split-mode", "layer",
        "--tensor-split", gpu_split,
        "--jinja",
        "--reasoning", args.reasoning,
        "--reasoning-format", args.reasoning_format,
        "--flash-attn", "on",
    ]
    if args.api_key:
        command += ["--api-key", args.api_key]
    if want_dspark and draft_path is not None:
        command += [
            "--model-draft", str(draft_path),
            "--spec-type", args.spec_type,
            "--spec-draft-n-max", str(args.spec_draft_n_max),
            "--n-gpu-layers-draft", str(args.n_gpu_layers_draft),
        ]
    # --fit defaults to "off" (DSpark README recommendation): with --fit on
    # llama.cpp cannot measure draft memory and skips the reserve, leaving the
    # ~11GB drafter outside the budget. --fit off pins placement explicitly.
    command += ["--fit", args.fit]
    # KV cache quantization defaults to q8_0/q8_0 (verified on this build).
    # Pass --cache-type-k "" --cache-type-v "" to fall back to f16.
    if args.cache_type_k:
        command += ["--cache-type-k", args.cache_type_k]
    if args.cache_type_v:
        command += ["--cache-type-v", args.cache_type_v]

    print("\n服务将运行于 http://0.0.0.0:%d/v1/chat/completions" % args.port)
    print("$ " + " ".join(command), flush=True)

    log_path = args.log_file
    if log_path is None:
        log_dir = args.log_dir.expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = log_dir / f"llama-server-{args.port}-{timestamp}.log"
    else:
        log_path = log_path.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"日志将写入：{log_path}", flush=True)

    require("tee")
    tee_flags = "-a" if not args.no_log_append else ""
    quoted_command = " ".join(shlex.quote(part) for part in command)
    quoted_log_path = shlex.quote(str(log_path))

    if args.no_console:
        shell_cmd = f"exec {quoted_command} >> {quoted_log_path} 2>&1"
    else:
        shell_cmd = (
            f"exec {quoted_command} 2>&1 | tee {tee_flags} {quoted_log_path}"
        )

    os.execvp("/bin/sh", ["/bin/sh", "-c", shell_cmd])


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"命令失败，退出码：{error.returncode}") from error

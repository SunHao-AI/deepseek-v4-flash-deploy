#!/usr/bin/env python3
"""engines/llamacpp.py — 官方 llama.cpp (GGUF) 适配器，含编译/下载/DSpark。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

from modelctl.core.capabilities import free_vram_total_mb
from modelctl.core.envfile import PROJECT_ROOT
from modelctl.engines._download import ensure_modelscope
from modelctl.engines.base import EngineAdapter, RequirementError

OFFICIAL_URL = "https://github.com/ggml-org/llama.cpp.git"
DSPARK_PATTERNS = ["*dspark*"]
CTX_PER_SLOT = 1_048_576


# --- 原样搬运自 start_v4_flash_gguf.py ---
def run(command: list[str], *, cwd: Path | None = None) -> None:
    """执行命令，失败时抛出 CalledProcessError。"""
    logger.info("\n$ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def require(name: str) -> None:
    """校验 PATH 中存在指定可执行文件，否则抛出 RequirementError。"""
    if shutil.which(name) is None:
        raise RequirementError(f"缺少 {name}。请安装后再运行脚本。")


def find_server(source: Path) -> Path:
    """定位编译产物 llama-server。

    找不到时返回预期路径 source/build/bin/llama-server（pre_start 会真正编译），
    避免在未编译环境下直接失败。
    """
    for candidate in (source / "build" / "bin" / "llama-server", source / "llama-server"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return source / "build" / "bin" / "llama-server"


def _find_first(destination: Path, patterns: list[str]) -> Path | None:
    """在 destination 下按 patterns 递归查找第一个匹配文件。"""
    for pattern in patterns:
        matches = sorted(destination.rglob(pattern))
        if matches:
            return matches[0]
    return None


def download_gguf(modelscope_id: str, model_root: Path, quant: str, want_dspark: bool) -> tuple[Path, Path | None]:
    """只下载指定量化版本的分片（需要时附带 DSpark 草稿）。

    返回 (模型首分片, 草稿路径或 None)。不会下载整个仓库。
    """
    ensure_modelscope()

    from modelscope import snapshot_download  # type: ignore[import-not-found]

    destination = model_root / modelscope_id.rsplit("/", 1)[-1]
    destination.mkdir(parents=True, exist_ok=True)

    patterns = [f"*{quant}*-00001-of-*.gguf", f"*{quant}*.gguf"]
    if want_dspark:
        patterns.extend(DSPARK_PATTERNS)

    logger.info(f"从 ModelScope 下载 {modelscope_id} 的 {quant} 分片（{', '.join(patterns)}）：{destination}")
    try:
        snapshot_download(
            model_id=modelscope_id,
            local_dir=str(destination),
            allow_file_pattern=patterns,
        )
    except Exception as error:
        raise RequirementError(
            f"ModelScope 下载失败。该 0731 GGUF 镜像可能尚未同步 {quant} 分片，或需要登录。\n"
            f"模型 ID：{modelscope_id}\n"
            "可尝试指定实际镜像 ID，或从可访问 Hugging Face 的机器下载后配置 model。"
        ) from error

    # ModelScope 保留仓库的 <quant>/ 子目录，分片不一定直接位于 destination 下。
    model_match = _find_first(destination, [f"*{quant}*-00001-of-*.gguf", f"*{quant}*.gguf"])
    if model_match is None:
        raise RequirementError(
            f"下载结束，但未找到 {quant} 的首分片：{destination}\n"
            "请检查 ModelScope 仓库中的实际量化名称；脚本没有下载官方 safetensors 权重。"
        )

    draft_match = None
    if want_dspark:
        draft_match = _find_first(destination, DSPARK_PATTERNS)
        if draft_match is None:
            logger.warning(
                f"\n警告：未在 {destination} 下找到 DSpark 草稿文件（{'、'.join(DSPARK_PATTERNS)}）。\n"
                "DSpark 将不会启用；请确认 ModelScope 镜像已同步 dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf，\n"
                "或改用 Hugging Face 手动下载后通过 draft 指定。\n"
            )
        else:
            logger.info(f"找到 DSpark 草稿：{draft_match}")

    return model_match, draft_match


class LlamaCppAdapter(EngineAdapter):
    def __init__(self, profile, caps):
        super().__init__(profile, caps)
        self._dspark = False
        self._draft: Path | None = None
        self._model: Path | None = None

    def check_requirements(self) -> None:
        cfg = self.profile.engine_config
        if self.caps.gpu_count == 0:
            raise RequirementError("未探测到 GPU（nvidia-smi 失败或无 GPU）")
        gpu_count = int(cfg.get("gpu_count", 8))
        if gpu_count > self.caps.gpu_count:
            raise RequirementError(f"profile gpu_count={gpu_count} 超过实际 GPU 数 {self.caps.gpu_count}")
        model = cfg.get("model")
        if not model:
            raise RequirementError(f"{self.profile.name}：llamacpp.model 必填")
        self._model = Path(model).expanduser()
        if not self._model.is_file() and not cfg.get("download"):
            raise RequirementError(f"找不到 GGUF 模型：{self._model}（且未配置 download 段）")
        # DSpark 草稿发现与显存降级
        if str(cfg.get("dspark", "on")).lower() in ("on", "true", "1"):
            self._draft = self._find_draft(cfg)
            if self._draft is None:
                self.warnings.append("未找到 DSpark 草稿模型，已自动关闭 DSpark")
            elif free_vram_total_mb(self.caps) < 11 * 1024:
                self.warnings.append("剩余显存不足 ~11GB，已自动关闭 DSpark")
                self._draft = None
            else:
                self._dspark = True
        # 显存预检：模型文件大小 × 1.1
        if self._model.is_file():
            need_mb = self._model.stat().st_size / 1024 / 1024 * 1.1
            if need_mb > free_vram_total_mb(self.caps):
                raise RequirementError(
                    f"剩余显存不足：模型约需 {need_mb:.0f}MB（×1.1），剩余 {free_vram_total_mb(self.caps)}MB"
                )

    def _find_draft(self, cfg: dict) -> Path | None:
        assert self._model is not None  # check_requirements 已设置模型路径
        if cfg.get("draft"):
            p = Path(cfg["draft"]).expanduser()
            return p if p.is_file() else None
        for base in (self._model.parent, *self._model.parents):
            if not base.exists():
                continue
            for name in ("dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", "dspark-DeepSeek-V4-Flash-0731-BF16.gguf"):
                if (base / name).is_file():
                    return base / name
            found = sorted(p for p in base.glob("*dspark*.gguf") if p.is_file())
            if found:
                return found[0]
        return None

    def build_command(self) -> tuple[list[str], dict[str, str]]:
        cfg = self.profile.engine_config
        assert self._model is not None  # check_requirements 已确保模型路径存在
        source = (
            Path(cfg.get("source_dir") or os.environ.get("LLAMACPP_SOURCE_DIR") or PROJECT_ROOT.parent / "llama.cpp")
            .expanduser()
            .resolve()
        )
        server = str(find_server(source))
        parallel = int(cfg.get("parallel", 2))
        ctx = int(cfg["ctx_size"]) if cfg.get("ctx_size") else parallel * CTX_PER_SLOT
        gpu_split = ",".join(["1"] * int(cfg.get("gpu_count", 8)))
        cmd = [
            server,
            "--model",
            str(self._model.resolve()),
            "--host",
            "0.0.0.0",
            "--port",
            str(self.profile.port),
            "--ctx-size",
            str(ctx),
            "--parallel",
            str(parallel),
            "--n-gpu-layers",
            "999",
            "--split-mode",
            "layer",
            "--tensor-split",
            gpu_split,
            "--jinja",
            "--reasoning",
            str(cfg.get("reasoning", "on")),
            "--reasoning-format",
            str(cfg.get("reasoning_format", "deepseek")),
            "--flash-attn",
            "on",
            "--metrics",
        ]
        cmd += self.api_key_args()
        if cfg.get("repeat_penalty"):
            cmd += ["--repeat-penalty", str(cfg["repeat_penalty"])]
        if self._dspark and self._draft is not None:
            cmd += [
                "--model-draft",
                str(self._draft),
                "--spec-type",
                str(cfg.get("spec_type", "draft-dspark")),
                "--spec-draft-n-max",
                str(cfg.get("spec_draft_n_max", 3)),
                "--n-gpu-layers-draft",
                str(cfg.get("n_gpu_layers_draft", 999)),
            ]
        cmd += ["--fit", str(cfg.get("fit", "off"))]
        if cfg.get("cache_type_k", "q8_0"):
            cmd += ["--cache-type-k", str(cfg.get("cache_type_k", "q8_0"))]
        if cfg.get("cache_type_v", "q8_0"):
            cmd += ["--cache-type-v", str(cfg.get("cache_type_v", "q8_0"))]
        env = {"MODELSCOPE_CACHE": os.environ["MODELSCOPE_CACHE"]} if os.environ.get("MODELSCOPE_CACHE") else {}
        return cmd, env

    def pre_start(self) -> None:
        cfg = self.profile.engine_config
        assert self._model is not None  # check_requirements 已确保模型路径存在
        source = (
            Path(cfg.get("source_dir") or os.environ.get("LLAMACPP_SOURCE_DIR") or PROJECT_ROOT.parent / "llama.cpp")
            .expanduser()
            .resolve()
        )
        if cfg.get("download") and not self._model.is_file():
            dl = cfg["download"]
            model_root = Path(os.environ.get("MODEL_ROOT") or PROJECT_ROOT.parent / "model-gguf")
            self._model, auto_draft = download_gguf(
                dl["modelscope_id"], model_root, dl.get("quant", "UD-Q8_K_XL"), self._dspark
            )
            if self._draft is None:
                self._draft = auto_draft
        require("git")
        require("cmake")
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--depth", "1", OFFICIAL_URL, str(source)])
        if not (source / "build" / "bin" / "llama-server").is_file():
            run(
                [
                    "cmake",
                    "-S",
                    str(source),
                    "-B",
                    str(source / "build"),
                    "-DGGML_CUDA=ON",
                    "-DCMAKE_BUILD_TYPE=Release",
                ]
            )
            run(["cmake", "--build", str(source / "build"), "--config", "Release", "-j"])

    def metrics_mapping(self) -> dict[str, list[str]]:
        return {
            "prompt_total": [
                "llamacpp:prompt_tokens_total",
                "llamacpp:tokens_evaluated_total",
                "llama_tokens_evaluated_total",
                "prompt_tokens_total",
            ],
            "predicted_total": [
                "llamacpp:tokens_predicted_total",
                "llamacpp:predicted_tokens_total",
                "llama_tokens_predicted_total",
                "tokens_predicted_total",
            ],
            "prompt_rate": ["llamacpp:prompt_tokens_seconds", "prompt_tokens_seconds"],
            "predicted_rate": [
                "llamacpp:predicted_tokens_seconds",
                "llamacpp:tokens_predicted_seconds",
                "predicted_tokens_seconds",
            ],
        }

    def stop_patterns(self) -> list[str]:
        return ["llama-server"]

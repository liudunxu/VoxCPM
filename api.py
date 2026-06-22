#!/usr/bin/env python3
"""Standalone VoxCPM HTTP API server.

API surface (paths, request format, response format) is kept consistent with the
OmniVoice ``api.py`` reference so that existing clients can switch backends with
minimal changes. The underlying synthesis is adapted to VoxCPM2's model API.

Endpoints:
  POST /api/voxcpm/synthesize   Synthesize audio with VoxCPM2
  GET  /api/health              Health check
  GET  /api/voxcpm/status       Model cache status
  POST /api/voxcpm/unload       Unload model from memory
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
from aiohttp import web

# Resolve the best device before importing the model so the import-time torch
# setup does not fight us.
from voxcpm import VoxCPM
from voxcpm.model.utils import resolve_runtime_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("voxcpm_api")

ROOT = Path(__file__).resolve().parent
WORK_ROOT = ROOT / "work"
OUTPUT_DIR = WORK_ROOT / "voxcpm_api_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TEXT_LEN = int(os.environ.get("VOXCPM_MAX_TEXT_LEN", "2000"))
MAX_REQUEST_MB = int(os.environ.get("VOXCPM_MAX_REQUEST_MB", "64"))
MAX_REQUEST_SIZE = MAX_REQUEST_MB * 1024 * 1024
VOXCPM_SEED_MOD = 2**31 - 1

_API_MODEL: Optional[VoxCPM] = None
_API_MODEL_ID = "openbmb/VoxCPM2"
_API_DEVICE: Optional[str] = None
_API_OPTIMIZE = False
_API_LOAD_DENOISER = True
_MODEL_LOAD_LOCK = asyncio.Lock()
_API_INFER_LOCK = asyncio.Lock()

# LRU cache of constructed request metadata used for stable seed derivation
# and dedup of identical outputs. Kept for parity with the OmniVoice server's
# caching hook.
_REQUEST_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_MAX_REQUEST_CACHE_SIZE = int(os.environ.get("VOXCPM_REQUEST_CACHE_SIZE", "128"))


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _json_response(data, status=200):
    return web.json_response(data, status=status)


def _error(message, status=400):
    return _json_response({"ok": False, "error": message}, status=status)


def _audio_duration_seconds(path):
    try:
        info = sf.info(str(path))
        if info.samplerate:
            return round(info.frames / info.samplerate, 3)
    except Exception:
        return None
    return None


def _decode_base64_audio_bytes(b64_data):
    """Decode base64 audio data to bytes. Supports data URI prefix."""
    b64_data = str(b64_data or "").strip()
    if b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1] if "," in b64_data else b64_data
    return base64.b64decode(b64_data)


def _write_base64_audio(b64_data, out_path):
    """Decode base64 audio data and write to file. Supports data URI prefix.

    Returns the written path.
    """
    audio_bytes = _decode_base64_audio_bytes(b64_data)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    return out_path


def _read_audio_base64(path):
    """Read audio file and return base64 encoded string with data URI prefix."""
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _relative_path(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path))


def _set_api_model(model, model_id, device, optimize, load_denoiser):
    global _API_MODEL, _API_MODEL_ID, _API_DEVICE, _API_OPTIMIZE, _API_LOAD_DENOISER
    _API_MODEL = model
    _API_MODEL_ID = model_id
    _API_DEVICE = device
    _API_OPTIMIZE = optimize
    _API_LOAD_DENOISER = load_denoiser


def _load_api_model_sync():
    logger.info(
        f"加载模型: {_API_MODEL_ID}, 设备: {_API_DEVICE}, "
        f"optimize: {_API_OPTIMIZE}, load_denoiser: {_API_LOAD_DENOISER} ..."
    )
    model = VoxCPM.from_pretrained(
        _API_MODEL_ID,
        device=_API_DEVICE,
        optimize=_API_OPTIMIZE,
        load_denoiser=_API_LOAD_DENOISER,
    )
    logger.info("模型加载完成！")
    return model


async def _ensure_api_model():
    global _API_MODEL
    if _API_MODEL is not None:
        return _API_MODEL
    async with _MODEL_LOAD_LOCK:
        # Double-check after acquiring lock.
        if _API_MODEL is not None:
            return _API_MODEL
        _API_MODEL = await asyncio.to_thread(_load_api_model_sync)
    return _API_MODEL


def _bool_option(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_seed(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value)) % VOXCPM_SEED_MOD
    except (TypeError, ValueError):
        return None


def _sha256_text(value):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _stable_seed_from_request(
    data, text, effective_prompt_text, reference_audio_base64, prompt_wav_base64
):
    explicit = _normalize_seed(data.get("seed") or data.get("voxcpm_seed"))
    if explicit is not None:
        return explicit
    if str(os.environ.get("VOXCPM_DETERMINISTIC", "1")).lower() in {"0", "false", "no", "off"}:
        return None
    payload = {
        "text": text,
        "prompt_text": effective_prompt_text,
        "reference_audio_sha256": _sha256_text(reference_audio_base64),
        "prompt_audio_sha256": _sha256_text(prompt_wav_base64),
        "model_id": data.get("model_id") or _API_MODEL_ID,
        "cfg_value": data.get("cfg_value", 2.0),
        "inference_timesteps": data.get("inference_timesteps", 10),
        "denoise": _bool_option(data.get("denoise"), True),
        "normalize": _bool_option(data.get("normalize"), False),
        "control_instruction": data.get("control_instruction") or data.get("instruct"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return int(digest[:12], 16) % VOXCPM_SEED_MOD


def _apply_seed(seed):
    seed = _normalize_seed(seed)
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    return seed


def _cleanup_temp_paths(*paths):
    for path in paths:
        if path is not None:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass


def _write_generated_audio(model, waveform, out_path):
    """Write a generated audio waveform (np.ndarray, shape (T,) or (C, T)) to disk."""
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu()
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()
    waveform = np.squeeze(waveform).astype(np.float32)
    sample_rate = int(model.tts_model.sample_rate)
    sf.write(str(out_path), waveform, sample_rate, subtype="PCM_16")
    return sample_rate


def _build_final_text(text: str, control_instruction: Optional[str]) -> str:
    """VoxCPM embeds voice-design / style control as a ``(control)text`` prefix.

    Any parentheses (half-width / full-width) are stripped from the control
    string to avoid breaking the ``(control)text`` prompt format.
    """
    control = (control_instruction or "").strip()
    control = re.sub(r"[()（）]", "", control).strip()
    if not control:
        return text
    return f"({control}){text}"


def _generate_voxcpm_audio(
    model,
    text,
    reference_wav_path=None,
    prompt_wav_path=None,
    prompt_text=None,
    cfg_value=2.0,
    inference_timesteps=10,
    normalize=False,
    denoise=False,
    min_len=2,
    max_len=4096,
    retry_badcase=True,
    retry_badcase_max_times=3,
    retry_badcase_ratio_threshold=6.0,
    seed=None,
):
    """Run model.generate and return the audio waveform (np.ndarray)."""
    kw: Dict[str, Any] = {
        "text": text.strip(),
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
        "normalize": bool(normalize),
        "denoise": bool(denoise),
        "min_len": int(min_len),
        "max_len": int(max_len),
        "retry_badcase": bool(retry_badcase),
        "retry_badcase_max_times": int(retry_badcase_max_times),
        "retry_badcase_ratio_threshold": float(retry_badcase_ratio_threshold),
    }
    if reference_wav_path:
        kw["reference_wav_path"] = reference_wav_path
    if prompt_wav_path:
        kw["prompt_wav_path"] = prompt_wav_path
    if prompt_text:
        kw["prompt_text"] = prompt_text

    _apply_seed(seed)
    return model.generate(**kw)


routes = web.RouteTableDef()


@routes.get("/api/health")
async def health(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    return _json_response({"ok": True, "service": "voxcpm2_api"})


@routes.get("/api/voxcpm/status")
@routes.get("/api/status")
async def status(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    cached_models = []
    if _API_MODEL is not None:
        cached_models.append({
            "model_id": _API_MODEL_ID,
            "device": _API_DEVICE,
            "optimize": _API_OPTIMIZE,
            "load_denoiser": _API_LOAD_DENOISER,
        })
    return _json_response({
        "ok": True,
        "models_cached": len(cached_models),
        "cached_models": cached_models,
    })


@routes.post("/api/voxcpm/unload")
@routes.post("/api/unload")
async def unload(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    global _API_MODEL
    count = 1 if _API_MODEL is not None else 0
    _API_MODEL = None
    _REQUEST_CACHE.clear()
    import gc
    gc.collect()
    if sys.platform != "win32":
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return _json_response({"ok": True, "unloaded": count})


@routes.post("/api/voxcpm/synthesize")
@routes.post("/api/synthesize")
async def synthesize(request):
    client_ip = request.remote or "-"
    req_id = uuid.uuid4().hex[:8]
    logger.info(f"[{req_id}] [{request.method}] {request.path} from {client_ip}")

    try:
        data = await request.json()
    except Exception as exc:
        tb = traceback.format_exc()
        logger.warning(f"[{req_id}] Failed to parse JSON: {exc}\n{tb}")
        return _error(f"Invalid JSON body: {exc}\n{tb}", status=400)

    text = re.sub(r"\s+", " ", (data.get("text") or "").strip())
    if not text:
        logger.warning(f"[{req_id}] Missing text parameter")
        return _error("text is required and cannot be empty.")
    if len(text) > MAX_TEXT_LEN:
        logger.warning(f"[{req_id}] Text too long: {len(text)} > {MAX_TEXT_LEN}")
        return _error(f"text exceeds max length {MAX_TEXT_LEN}.")

    reference_audio_base64 = data.get("reference_audio_base64")
    prompt_wav_base64 = (
        data.get("prompt_wav_base64")
        or data.get("prompt_audio_base64")
        or data.get("prompt_wav")
    )
    prompt_text = re.sub(r"\s+", " ", (data.get("prompt_text") or "").strip())
    effective_prompt_text = prompt_text if prompt_wav_base64 else ""

    # Control instruction drives VoxCPM's voice-design / controllable-cloning
    # style. Accept both ``control_instruction`` (VoxCPM naming) and ``instruct``
    # (OmniVoice naming) for compatibility.
    control_instruction = (
        data.get("control_instruction") or data.get("instruct") or ""
    )

    # Generation parameters.
    cfg_value = float(data.get("cfg_value", 2.0))
    inference_timesteps = int(data.get("inference_timesteps", 10))
    denoise = _bool_option(data.get("denoise"), True)
    normalize = _bool_option(data.get("normalize"), False)
    optimize = _bool_option(data.get("optimize"), False)
    min_len = int(data.get("min_len", 2))
    max_len = int(data.get("max_len", 4096))
    retry_badcase = _bool_option(data.get("retry_badcase"), True)
    retry_badcase_max_times = int(data.get("retry_badcase_max_times", 3))
    retry_badcase_ratio_threshold = float(data.get("retry_badcase_ratio_threshold", 6.0))
    seed = _stable_seed_from_request(
        data, text, effective_prompt_text, reference_audio_base64, prompt_wav_base64
    )

    # Prepare output directory.
    out_dir = Path(data.get("output_dir") or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decode reference / prompt audio to temp files.
    ref_temp_path = None
    prompt_temp_path = None
    resolved_ref = None
    resolved_prompt = None
    ref_duration = None

    if reference_audio_base64:
        ref_temp_path = out_dir / f"ref_{uuid.uuid4().hex}.wav"
        ref_temp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ref_audio_bytes = _decode_base64_audio_bytes(reference_audio_base64)
            ref_temp_path.write_bytes(ref_audio_bytes)
            resolved_ref = str(ref_temp_path)
            ref_duration = _audio_duration_seconds(ref_temp_path)
            logger.info(
                f"[{req_id}] reference audio decoded: {ref_temp_path} "
                f"({ref_temp_path.stat().st_size} bytes), duration={ref_duration}s"
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"[{req_id}] Failed to decode reference_audio_base64: {exc}\n{tb}")
            if ref_temp_path and ref_temp_path.exists():
                ref_temp_path.unlink(missing_ok=True)
            return _error(f"Failed to decode reference_audio_base64: {exc}\n{tb}")

    if prompt_wav_base64:
        prompt_temp_path = out_dir / f"prompt_{uuid.uuid4().hex}.wav"
        try:
            prompt_audio_bytes = _decode_base64_audio_bytes(prompt_wav_base64)
            prompt_temp_path.write_bytes(prompt_audio_bytes)
            resolved_prompt = str(prompt_temp_path)
            logger.info(
                f"[{req_id}] prompt wav decoded: {prompt_temp_path} "
                f"({prompt_temp_path.stat().st_size} bytes)"
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"[{req_id}] Failed to decode prompt_wav_base64: {exc}\n{tb}")
            _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
            return _error(f"Failed to decode prompt_wav_base64: {exc}\n{tb}")

    logger.info(
        f"[{req_id}] params: text_len={len(text)}, has_ref={bool(reference_audio_base64)}, "
        f"ref_duration={ref_duration}s, has_prompt_wav={bool(prompt_wav_base64)}, "
        f"prompt_len={len(effective_prompt_text)}, "
        f"control_instruction={'yes' if control_instruction else 'no'}, "
        f"requested_model={data.get('model_id') or ''}, loaded_model={_API_MODEL_ID}, "
        f"device={_API_DEVICE}, cfg={cfg_value}, steps={inference_timesteps}, "
        f"denoise={denoise}, normalize={normalize}, optimize={optimize}, "
        f"retry_badcase={retry_badcase}, seed={seed if seed is not None else '-'}"
    )

    out_name = data.get("output_name")
    if not out_name:
        key = hashlib.sha256(
            json.dumps({
                "text": text,
                "prompt": effective_prompt_text,
                "control": control_instruction,
                "ref_b64_len": len(reference_audio_base64) if reference_audio_base64 else 0,
                "prompt_b64_len": len(prompt_wav_base64) if prompt_wav_base64 else 0,
                "model": _API_MODEL_ID,
                "device": _API_DEVICE,
                "cfg": cfg_value,
                "steps": inference_timesteps,
                "denoise": denoise,
                "normalize": normalize,
                "optimize": optimize,
                "seed": seed,
            }, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        out_name = f"voxcpm_{key}.wav"
    out_path = out_dir / out_name

    # Load model (loading is serialized via _MODEL_LOAD_LOCK).
    model = await _ensure_api_model()

    # Determine the generation mode for logging.
    if resolved_prompt and effective_prompt_text:
        mode = "Ultimate Cloning (prompt_wav + prompt_text)"
    elif resolved_ref:
        mode = "Controllable Cloning (reference_wav)"
    elif control_instruction:
        mode = "Voice Design (control instruction)"
    else:
        mode = "Default TTS"

    final_text = _build_final_text(text, control_instruction)

    start_time = time.time()
    try:
        async with _API_INFER_LOCK:
            logger.info(f"[{req_id}] [{mode}] synthesis started -> {out_path}")
            audio_waveform = await asyncio.to_thread(
                _generate_voxcpm_audio,
                model,
                final_text,
                reference_wav_path=resolved_ref,
                prompt_wav_path=resolved_prompt,
                prompt_text=effective_prompt_text or None,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
                min_len=min_len,
                max_len=max_len,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                seed=seed,
            )
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Synthesis failed: {exc}\n{tb}")
        _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
        return _error(f"Synthesis failed: {exc}\n{tb}", status=502)

    try:
        sample_rate = _write_generated_audio(model, audio_waveform, out_path)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Failed to write output audio: {exc}\n{tb}")
        _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
        return _error(f"Failed to write output audio: {exc}\n{tb}", status=502)

    elapsed = round(time.time() - start_time, 3)
    if not out_path.exists():
        logger.error(f"[{req_id}] Output file not created: {out_path}")
        _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
        return _error("Synthesis finished but output file was not created.", status=502)

    audio_duration = _audio_duration_seconds(out_path)
    logger.info(
        f"[{req_id}] synthesis finished in {elapsed}s, output: {out_path} "
        f"({out_path.stat().st_size} bytes), audio_duration={audio_duration}, "
        f"sample_rate={sample_rate}"
    )

    try:
        output_base64 = _read_audio_base64(out_path)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Failed to encode output audio: {exc}\n{tb}")
        _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
        return _error(f"Failed to encode output audio: {exc}\n{tb}", status=502)

    _cleanup_temp_paths(ref_temp_path, prompt_temp_path)

    logger.info(f"[{req_id}] response sent, audio_base64_len={len(output_base64)}")
    return _json_response({
        "ok": True,
        "audio_base64": output_base64,
        "output_path": str(out_path.resolve()),
        "relative_path": _relative_path(out_path),
        "elapsed_seconds": elapsed,
        "audio_duration_seconds": audio_duration,
        "sample_rate": sample_rate,
        "seed": seed,
        "mode": mode,
        "generation_params": {
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "denoise": denoise,
            "normalize": normalize,
            "min_len": min_len,
            "max_len": max_len,
            "retry_badcase": retry_badcase,
            "retry_badcase_max_times": retry_badcase_max_times,
            "retry_badcase_ratio_threshold": retry_badcase_ratio_threshold,
        },
        "duration_match": {
            "ref_duration": ref_duration,
            "actual_duration": audio_duration,
            "match_ratio": round(audio_duration / ref_duration, 3) if ref_duration and audio_duration else None,
        },
    })


@routes.get("/")
async def index(request):
    return web.Response(
        content_type="text/html",
        text="""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>VoxCPM API</title></head>
<body>
  <h1>VoxCPM2 API Server</h1>
  <pre>
GET  /api/health
GET  /api/voxcpm/status  (alias: /api/status)
POST /api/voxcpm/unload  (alias: /api/unload)
POST /api/voxcpm/synthesize  (alias: /api/synthesize)

Request (JSON):
{
  "text": "要合成的文本",
  "reference_audio_base64": "data:audio/wav;base64,xxxx...",  // 可选；声音克隆参考音频
  "prompt_wav_base64": "data:audio/wav;base64,yyyy...",       // 可选；极致克隆模式参考音频
  "prompt_text": "参考音频对应的文本",                          // 可选；仅随 prompt_wav 使用（极致克隆）
  "control_instruction": "年轻女性，温柔甜美",                  // 可选；声音设计 / 可控克隆风格描述（别名 instruct）
  "model_id": "openbmb/VoxCPM2",
  "device": "auto",                     // auto / cpu / mps / cuda / cuda:N
  "cfg_value": 2.0,                     // 可选；CFG 引导强度
  "inference_timesteps": 10,            // 可选；LocDiT 流匹配步数
  "denoise": true,                      // 可选；是否对参考音频降噪（需加载 denoiser）
  "normalize": false,                   // 可选；文本规范化
  "optimize": false,                    // 可选；torch.compile 优化（仅 CUDA）
  "min_len": 2,                         // 可选；最小音频长度
  "max_len": 4096,                      // 可选；最大 token 长度
  "retry_badcase": true,                // 可选；检测 badcase 自动重试
  "retry_badcase_max_times": 3,         // 可选；badcase 最大重试次数
  "retry_badcase_ratio_threshold": 6.0, // 可选；音文比阈值
  "seed": 123456789,                    // 可选；不传时默认派生稳定 seed
  "output_dir": "",                     // 可选；自定义输出目录
  "output_name": ""                     // 可选；自定义输出文件名
}

生成模式（由参数自动判定）：
  - 极致克隆：prompt_wav_base64 + prompt_text（完整还原音色细节）
  - 可控克隆：reference_audio_base64（可选叠加 control_instruction 控制风格）
  - 声音设计：仅 control_instruction（无需参考音频，从描述创造声音）
  - 默认 TTS：仅 text

Response (JSON):
{
  "ok": true,
  "audio_base64": "data:audio/wav;base64,xxxx...",
  "output_path": "/abs/path/to/output.wav",
  "relative_path": "work/voxcpm_api_outputs/voxcpm_xxx.wav",
  "elapsed_seconds": 12.345,
  "audio_duration_seconds": 2.431,
  "sample_rate": 48000,
  "seed": 123456789,
  "mode": "Controllable Cloning (reference_wav)",
  "generation_params": {
    "cfg_value": 2.0,
    "inference_timesteps": 10,
    "denoise": true,
    "normalize": false,
    "min_len": 2,
    "max_len": 4096,
    "retry_badcase": true,
    "retry_badcase_max_times": 3,
    "retry_badcase_ratio_threshold": 6.0
  },
  "duration_match": {
    "ref_duration": 3.5,
    "actual_duration": 2.431,
    "match_ratio": 0.695
  }
}
  </pre>
</body>
</html>""",
    )


async def on_startup(app):
    print(
        f"[VoxCPM API] listening on http://{app['host']}:{app['port']} "
        f"(max request {MAX_REQUEST_MB} MB)"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="VoxCPM API")
    parser.add_argument(
        "--model", default="openbmb/VoxCPM2",
        help="模型路径或 HuggingFace 仓库 ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--device", default=None, help="运行设备 (auto/cuda/mps/cpu)")
    parser.add_argument("--ip", default="0.0.0.0", help="服务器 IP")
    parser.add_argument("--port", type=int, default=6006, help="服务器端口")
    parser.add_argument(
        "--optimize", action="store_true",
        help="启用 torch.compile 优化（仅 CUDA 有效）",
    )
    parser.add_argument(
        "--no-denoiser", action="store_true",
        help="不加载 ZipEnhancer 降噪模型（默认加载）",
    )
    args = parser.parse_args(argv)

    device = args.device or get_best_device()
    device = resolve_runtime_device(device, "cuda")
    _set_api_model(
        None, args.model, device,
        optimize=args.optimize,
        load_denoiser=not args.no_denoiser,
    )

    app = web.Application(client_max_size=MAX_REQUEST_SIZE)
    app["host"] = args.ip
    app["port"] = args.port
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    web.run_app(app, host=args.ip, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())

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
from typing import Any, Dict, Optional, Tuple

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

# LRU cache of denoised reference audio files, keyed by audio content hash.
# The denoised file is reused across requests that pass the same reference,
# avoiding repeated ZipEnhancer passes on identical uploads.
_REF_DENOISE_CACHE: "OrderedDict[str, Path]" = OrderedDict()
_MAX_REF_DENOISE_CACHE_SIZE = int(os.environ.get("VOXCPM_REF_CACHE_SIZE", "32"))

# LRU cache of VoxCPM prompt caches (build_prompt_cache results), keyed by the
# denoised reference audio hash + prompt metadata. build_prompt_cache encodes
# the reference/prompt audio through the AudioVAE, so reusing it skips the most
# expensive per-request clone setup when the same reference is submitted again.
_PROMPT_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_MAX_PROMPT_CACHE_SIZE = int(os.environ.get("VOXCPM_PROMPT_CACHE_SIZE", "32"))

# Session-scoped prompt caches for chained continuation across dubbing segments.
# Each session holds a growing prompt cache (ref + accumulated generated audio)
# so segment N+1 continues seamlessly from segment N's voice. Sessions expire
# after a period of inactivity to bound memory.
_SESSION_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_MAX_SESSION_COUNT = int(os.environ.get("VOXCPM_MAX_SESSIONS", "64"))
_SESSION_TTL_SECONDS = int(os.environ.get("VOXCPM_SESSION_TTL", "1800"))


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


def _lru_put(cache: OrderedDict, max_size: int, key: str, value: Any) -> None:
    """Insert/refresh a key in an OrderedDict-backed LRU cache."""
    if key in cache:
        cache.move_to_end(key)
        cache[key] = value
        return
    if len(cache) >= max_size:
        cache.popitem(last=False)
    cache[key] = value


def _denoise_reference_cached(
    model, audio_bytes: bytes, out_dir: Path, denoise: bool
) -> Tuple[Path, bool]:
    """Return a (possibly denoised) reference audio path, caching by content hash.

    Returns (path, was_denoised). When denoise is False or no denoiser is
    loaded, a fresh temp file holding the raw bytes is returned (not cached).
    """
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    if denoise and getattr(model, "denoiser", None) is not None:
        cached = _REF_DENOISE_CACHE.get(audio_hash)
        if cached is not None and cached.exists():
            _REF_DENOISE_CACHE.move_to_end(audio_hash)
            return cached, True
        out_path = out_dir / f"ref_den_{audio_hash[:16]}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / f"ref_raw_{uuid.uuid4().hex}.wav"
        try:
            raw_path.write_bytes(audio_bytes)
            model.denoiser.enhance(str(raw_path), output_path=str(out_path))
            _cleanup_temp_paths(raw_path)
            _lru_put(_REF_DENOISE_CACHE, _MAX_REF_DENOISE_CACHE_SIZE, audio_hash, out_path)
            return out_path, True
        except Exception:
            _cleanup_temp_paths(raw_path, out_path)
            # Fall through to raw handling below.
            pass

    out_path = out_dir / f"ref_{uuid.uuid4().hex}.wav"
    out_path.write_bytes(audio_bytes)
    return out_path, False


def _build_prompt_cache_cached(
    model,
    audio_path: str,
    prompt_text: Optional[str],
    has_reference: bool,
    has_prompt: bool,
    trim_silence_vad: bool,
    audio_hash: str,
):
    """Build (and cache) VoxCPM's prompt cache for a reference/prompt combo.

    Mirrors core._generate's clone setup but caches the result so repeated
    clones of the same reference skip the AudioVAE encode. Only the actual
    encode is cached; generation itself still runs fresh per request.
    """
    tts = model.tts_model
    is_v2 = type(tts).__name__ == "VoxCPM2Model"
    if has_reference and not is_v2:
        raise ValueError("reference_wav_path is only supported with VoxCPM2 models")

    cache_key = hashlib.sha256(
        json.dumps({
            "audio_hash": audio_hash,
            "prompt_text": prompt_text or "",
            "has_reference": has_reference,
            "has_prompt": has_prompt,
            "trim_silence_vad": trim_silence_vad,
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    cached = _PROMPT_CACHE.get(cache_key)
    if cached is not None:
        _PROMPT_CACHE.move_to_end(cache_key)
        return cached, cache_key

    if is_v2:
        prompt_cache = tts.build_prompt_cache(
            prompt_text=prompt_text,
            prompt_wav_path=audio_path if has_prompt else None,
            reference_wav_path=audio_path if has_reference else None,
            trim_silence_vad=trim_silence_vad,
        )
    else:
        prompt_cache = tts.build_prompt_cache(
            prompt_text=prompt_text,
            prompt_wav_path=audio_path if has_prompt else None,
        )
    _lru_put(_PROMPT_CACHE, _MAX_PROMPT_CACHE_SIZE, cache_key, prompt_cache)
    return prompt_cache, cache_key


def _measure_silence_ratio(waveform, threshold: float = 0.01) -> float:
    arr = np.asarray(waveform)
    if arr.size == 0:
        return 1.0
    return float(np.mean(np.abs(arr) < threshold))


def _compute_rms(waveform) -> float:
    arr = np.asarray(waveform)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def _check_audio_quality(waveform, sample_rate: int = 48000) -> list:
    """Detect common badcase patterns the model's built-in retry does not cover.

    VoxCPM's retry_badcase only flags over-long output (audio/text ratio). It
    does not catch silence-heavy, clipped, or too-quiet audio, so we add those
    here as a signal for a seed-based retry.
    """
    issues = []
    arr = np.asarray(waveform).astype(np.float32).squeeze()
    if arr.size == 0 or arr.shape[-1] / max(sample_rate, 1) < 0.05:
        issues.append("empty")
    if _measure_silence_ratio(arr) > 0.5:
        issues.append("too_much_silence")
    if arr.size > 0 and float(np.abs(arr).max()) > 0.99:
        issues.append("clipping")
    rms = _compute_rms(arr)
    if 0 < rms < 0.005:
        issues.append("too_quiet")
    return issues


# Time-stretch limits for duration alignment. Beyond these ratios, stretching
# degrades audibly (warbling / artifacts) and we fall back to the raw output.
TIME_STRETCH_MIN_RATE = 0.70   # fastest: 1.43x stretch (slow down)
TIME_STRETCH_MAX_RATE = 1.43   # slowest: 0.70x compress (speed up)


def _time_stretch_to_duration(
    waveform, sample_rate: int, target_duration: Optional[float],
    tolerance: float = 0.05,
) -> Tuple[np.ndarray, Optional[dict]]:
    """Pitch-preserving time stretch to hit ``target_duration`` (seconds).

    Uses librosa.effects.time_stretch (phase vocoder). Only stretches when the
    deviation exceeds ``tolerance`` seconds AND the required rate is within the
    safe band; otherwise returns the waveform unchanged.

    Returns (waveform, info) where info describes what was done (or None).
    """
    if target_duration is None or target_duration <= 0:
        return waveform, None
    arr = np.asarray(waveform).astype(np.float32).squeeze()
    if arr.size == 0:
        return waveform, None

    actual = arr.shape[-1] / sample_rate
    if abs(actual - target_duration) <= tolerance:
        return waveform, None

    rate = actual / target_duration  # >1 => too long, speed up
    if not (TIME_STRETCH_MIN_RATE <= rate <= TIME_STRETCH_MAX_RATE):
        return waveform, {
            "applied": False,
            "reason": "out_of_safe_range",
            "required_rate": round(rate, 3),
        }

    import librosa
    stretched = librosa.effects.time_stretch(arr, rate=rate)
    info = {
        "applied": True,
        "actual_duration": round(actual, 3),
        "target_duration": round(target_duration, 3),
        "rate": round(rate, 3),
        "stretched_duration": round(stretched.shape[-1] / sample_rate, 3),
    }
    return stretched.astype(np.float32), info


def _session_get(session_id: str, model) -> Optional[dict]:
    """Look up (and refresh) a session prompt cache. Returns None if missing."""
    if not session_id:
        return None
    entry = _SESSION_CACHE.get(session_id)
    if entry is None:
        return None
    entry["last_used"] = time.time()
    _SESSION_CACHE.move_to_end(session_id)
    return entry


def _session_put(session_id: str, model, prompt_cache: dict, ref_audio_hash: str,
                 accumulated_text: str) -> None:
    """Create or update a session prompt cache entry."""
    entry = {
        "prompt_cache": prompt_cache,
        "ref_audio_hash": ref_audio_hash,
        "accumulated_text": accumulated_text,
        "created_at": time.time(),
        "last_used": time.time(),
    }
    _lru_put(_SESSION_CACHE, _MAX_SESSION_COUNT, session_id, entry)


def _session_evict_expired() -> int:
    """Drop sessions idle longer than the TTL. Returns count evicted."""
    if not _SESSION_CACHE:
        return 0
    now = time.time()
    expired = [k for k, v in _SESSION_CACHE.items()
               if now - v["last_used"] > _SESSION_TTL_SECONDS]
    for k in expired:
        _SESSION_CACHE.pop(k, None)
    return len(expired)


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
    trim_silence_vad=True,
    seed=None,
    quality_retry=True,
    quality_retry_max=2,
    session_id=None,
    session_continue=False,
):
    """Run generation via the cached prompt-cache path.

    Differences vs. a plain ``model.generate`` call:
      * Builds (and LRU-caches) the prompt cache, skipping the AudioVAE encode
        on repeated clones of the same reference.
      * Enables ``trim_silence_vad`` for cloning modes to improve alignment.
      * After generation, runs a quality check and retries with a fresh seed
        if silence/clipping/quiet badcases are detected (the model's built-in
        retry only catches over-long output).
      * When ``session_id`` is provided, uses a session-scoped prompt cache so
        segment N+1 continues from segment N's generated audio (chained voice
        continuity for dubbing). The generated audio features are returned so
        the caller can merge them back into the session.

    Returns:
        (waveform_np, attempts_log, quality_issues, quality_retried,
         prompt_cache_hit, pred_audio_feat, session_info)
    """
    tts = model.tts_model
    has_reference = bool(reference_wav_path)
    has_prompt = bool(prompt_wav_path) and bool(prompt_text)

    # Determine the prompt cache to use: session cache (for continuation) takes
    # precedence, then the freshly-built reference/prompt cache.
    prompt_cache = None
    prompt_cache_hit = False
    audio_hash = ""
    session_info = {"used": False}

    session_entry = _session_get(session_id, model) if session_id else None
    if session_continue and session_entry is not None:
        prompt_cache = session_entry["prompt_cache"]
        prompt_cache_hit = True
        session_info = {
            "used": True,
            "continued": True,
            "ref_audio_hash": session_entry["ref_audio_hash"],
            "accumulated_text_len": len(session_entry["accumulated_text"]),
        }
        logger.info(
            f"[session] continuing session={session_id} "
            f"(acc_text_len={len(session_entry['accumulated_text'])})"
        )
    elif has_reference or has_prompt:
        clone_path = reference_wav_path or prompt_wav_path
        try:
            import hashlib as _h
            audio_hash = _h.sha256(Path(clone_path).read_bytes()).hexdigest()
        except Exception:
            audio_hash = str(clone_path)
        prompt_cache, prompt_cache_hit = _build_prompt_cache_cached(
            model,
            audio_path=clone_path,
            prompt_text=prompt_text if has_prompt else None,
            has_reference=has_reference,
            has_prompt=has_prompt,
            trim_silence_vad=trim_silence_vad,
            audio_hash=audio_hash,
        )

    final_text = text.strip()
    if normalize:
        if getattr(model, "text_normalizer", None) is None:
            try:
                from voxcpm.utils.text_normalize import TextNormalizer
                model.text_normalizer = TextNormalizer()
            except Exception as exc:
                logger.warning(f"text normalization unavailable: {exc}")
        if getattr(model, "text_normalizer", None) is not None:
            final_text = model.text_normalizer.normalize(final_text)

    def _gen(seed_value):
        _apply_seed(seed_value)
        wav, _text_tok, pred_feat = tts.generate_with_prompt_cache(
            target_text=final_text,
            prompt_cache=prompt_cache,
            min_len=int(min_len),
            max_len=int(max_len),
            inference_timesteps=int(inference_timesteps),
            cfg_value=float(cfg_value),
            retry_badcase=bool(retry_badcase),
            retry_badcase_max_times=int(retry_badcase_max_times),
            retry_badcase_ratio_threshold=float(retry_badcase_ratio_threshold),
        )
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu()
        if hasattr(wav, "numpy"):
            wav = wav.numpy()
        return np.squeeze(wav).astype(np.float32), pred_feat

    attempt_log = []
    quality_issues = []
    quality_retried = False

    base_seed = _normalize_seed(seed)
    sample_rate = int(tts.sample_rate)
    wav, pred_feat = _gen(base_seed)
    issues = _check_audio_quality(wav, sample_rate)
    attempt_log.append({"attempt": 1, "seed": base_seed, "quality_issues": issues})

    if issues and quality_retry:
        for i in range(quality_retry_max):
            if not issues:
                break
            logger.info(f"Quality issues on attempt {i + 1}: {issues}; retrying with a fresh seed")
            quality_retried = True
            next_seed = (base_seed + i + 1) % VOXCPM_SEED_MOD if base_seed is not None else None
            wav2, pred_feat2 = _gen(next_seed)
            issues2 = _check_audio_quality(wav2, sample_rate)
            attempt_log.append({"attempt": i + 2, "seed": next_seed, "quality_issues": issues2})
            # Keep whichever has fewer issues; prefer the kept one's features.
            if len(issues2) < len(issues):
                wav, issues, pred_feat = wav2, issues2, pred_feat2
            if not issues:
                break
    quality_issues = issues

    # If a session is active, merge this segment's audio back so the next call
    # in the session continues from it. We move pred_feat to CPU to keep the
    # session cache device-agnostic and avoid pinning GPU memory.
    if session_id is not None:
        feat_cpu = pred_feat.detach().cpu() if hasattr(pred_feat, "detach") else pred_feat
        if session_entry is not None and session_continue:
            # Existing session: extend accumulated audio + text.
            merged = tts.merge_prompt_cache(
                session_entry["prompt_cache"], final_text, feat_cpu
            )
            session_entry["prompt_cache"] = merged
            session_entry["accumulated_text"] += final_text
            session_entry["last_used"] = time.time()
            session_info["merged"] = True
            session_info["accumulated_text_len"] = len(session_entry["accumulated_text"])
        elif prompt_cache is not None:
            # New session seeded from this segment's reference/prompt cache.
            merged = tts.merge_prompt_cache(prompt_cache, final_text, feat_cpu)
            _session_put(
                session_id, model, merged, audio_hash, final_text
            )
            session_info = {
                "used": True,
                "continued": False,
                "created": True,
                "accumulated_text_len": len(final_text),
            }

    return wav, attempt_log, quality_issues, quality_retried, prompt_cache_hit, pred_feat, session_info


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
    _REF_DENOISE_CACHE.clear()
    _PROMPT_CACHE.clear()
    _SESSION_CACHE.clear()
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
    trim_silence_vad = _bool_option(data.get("trim_silence_vad"), True)
    quality_retry = _bool_option(data.get("quality_retry"), True)
    quality_retry_max = int(data.get("quality_retry_max", 2))

    # Session-based chained continuation for dubbing: pass the same session_id
    # across segments of one voice to make segment N+1 continue from segment N.
    # The first call seeds the session; subsequent calls set session_continue.
    session_id = (data.get("session_id") or "").strip() or None
    session_continue = _bool_option(data.get("session_continue"), True)

    # Duration alignment (dubbing): target_duration_ms is the desired output
    # length; we time-stretch the generated audio to hit it (pitch-preserving).
    target_duration_ms = data.get("target_duration_ms")
    duration_tolerance_ms = data.get("duration_tolerance_ms")

    # Allow the caller to force a specific seed (per-character fixed seed for
    # cross-segment voice consistency). Falls back to the derived stable seed.
    explicit_seed = _normalize_seed(data.get("seed") or data.get("voxcpm_seed"))
    seed = explicit_seed if explicit_seed is not None else _stable_seed_from_request(
        data, text, effective_prompt_text, reference_audio_base64, prompt_wav_base64
    )

    # Prepare output directory.
    out_dir = Path(data.get("output_dir") or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Opportunistically evict expired dubbing sessions.
    if _SESSION_CACHE:
        evicted = _session_evict_expired()
        if evicted:
            logger.info(f"[{req_id}] evicted {evicted} expired sessions")

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

            # Produce (cached) denoised reference/prompt paths. The denoiser
            # pass is the slowest part of clone setup; caching by content hash
            # lets repeat uploads reuse it. build_prompt_cache (AudioVAE encode)
            # is then cached on top of these stable paths.
            managed_ref = None
            managed_prompt = None
            try:
                if resolved_ref:
                    managed_ref, ref_denoised = _denoise_reference_cached(
                        model, ref_audio_bytes, out_dir, denoise
                    )
                    logger.info(f"[{req_id}] ref path for clone: {managed_ref} (denoised={ref_denoised})")
                if resolved_prompt:
                    managed_prompt, prompt_denoised = _denoise_reference_cached(
                        model, prompt_audio_bytes, out_dir, denoise
                    )
                    logger.info(f"[{req_id}] prompt path for clone: {managed_prompt} (denoised={prompt_denoised})")

                (
                    audio_waveform,
                    attempt_log,
                    quality_issues,
                    quality_retried,
                    prompt_cache_hit,
                    _pred_feat,
                    session_info,
                ) = await asyncio.to_thread(
                    _generate_voxcpm_audio,
                    model,
                    final_text,
                    reference_wav_path=str(managed_ref) if managed_ref else None,
                    prompt_wav_path=str(managed_prompt) if managed_prompt else None,
                    prompt_text=effective_prompt_text or None,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    normalize=normalize,
                    denoise=False,  # already denoised above via cache
                    min_len=min_len,
                    max_len=max_len,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=retry_badcase_max_times,
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                    trim_silence_vad=trim_silence_vad,
                    seed=seed,
                    quality_retry=quality_retry,
                    quality_retry_max=quality_retry_max,
                    session_id=session_id,
                    session_continue=session_continue,
                )
            finally:
                # Clean up only the raw (non-cached) temp files. Cached denoised
                # files live in _REF_DENOISE_CACHE and must persist.
                for p in (managed_ref, managed_prompt):
                    if p is not None and p not in _REF_DENOISE_CACHE.values():
                        _cleanup_temp_paths(p)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Synthesis failed: {exc}\n{tb}")
        _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
        return _error(f"Synthesis failed: {exc}\n{tb}", status=502)

    # Duration alignment: pitch-preserving time-stretch toward target_duration.
    stretch_info = None
    target_duration_sec = (
        float(target_duration_ms) / 1000.0 if target_duration_ms is not None else None
    )
    tolerance_sec = (
        float(duration_tolerance_ms) / 1000.0 if duration_tolerance_ms is not None else 0.05
    )
    if target_duration_sec is not None:
        sample_rate_pre = int(model.tts_model.sample_rate)
        audio_waveform, stretch_info = _time_stretch_to_duration(
            audio_waveform, sample_rate_pre, target_duration_sec, tolerance=tolerance_sec
        )
        if stretch_info:
            logger.info(f"[{req_id}] time-stretch: {stretch_info}")

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
        f"sample_rate={sample_rate}, quality_issues={quality_issues}, "
        f"quality_retried={quality_retried}, prompt_cache_hit={prompt_cache_hit}, "
        f"session={'on' if session_info.get('used') else 'off'}"
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
        "quality_issues": quality_issues,
        "quality_retried": quality_retried,
        "quality_attempt_log": attempt_log,
        "prompt_cache_hit": prompt_cache_hit,
        "session": session_info,
        "duration_alignment": {
            "target_duration_ms": target_duration_ms,
            "duration_tolerance_ms": duration_tolerance_ms,
            "stretch": stretch_info,
            "actual_duration_seconds": audio_duration,
            "target_duration_seconds": round(target_duration_sec, 3) if target_duration_sec else None,
        },
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
            "trim_silence_vad": trim_silence_vad,
            "quality_retry": quality_retry,
            "quality_retry_max": quality_retry_max,
            "session_id": session_id,
            "session_continue": session_continue,
        },
        "duration_match": {
            "ref_duration": ref_duration,
            "actual_duration": audio_duration,
            "match_ratio": round(audio_duration / ref_duration, 3) if ref_duration and audio_duration else None,
        },
    })


@routes.get("/api/voxcpm/sessions")
@routes.get("/api/sessions")
async def sessions_list(request):
    """List active dubbing sessions (id + accumulated text length + last used)."""
    now = time.time()
    items = []
    for sid, entry in _SESSION_CACHE.items():
        items.append({
            "session_id": sid,
            "ref_audio_hash": entry.get("ref_audio_hash", "")[:16],
            "accumulated_text_len": len(entry.get("accumulated_text", "")),
            "idle_seconds": round(now - entry["last_used"], 1),
        })
    return _json_response({
        "ok": True,
        "count": len(items),
        "sessions": items,
        "ttl_seconds": _SESSION_TTL_SECONDS,
        "max_sessions": _MAX_SESSION_COUNT,
    })


@routes.post("/api/voxcpm/session/clear")
@routes.post("/api/session/clear")
async def session_clear(request):
    """Clear one session (body: {"session_id": "..."}) or all sessions if none given."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    sid = (data.get("session_id") or "").strip()
    if sid:
        removed = 1 if _SESSION_CACHE.pop(sid, None) is not None else 0
        return _json_response({"ok": True, "cleared": removed, "scope": "one", "session_id": sid})
    removed = len(_SESSION_CACHE)
    _SESSION_CACHE.clear()
    return _json_response({"ok": True, "cleared": removed, "scope": "all"})


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
GET  /api/voxcpm/sessions  (alias: /api/sessions)            // 列出配音会话
POST /api/voxcpm/session/clear  (alias: /api/session/clear)  // 清除会话

Request (JSON):
{
  "text": "要合成的文本",
  "reference_audio_base64": "data:audio/wav;base64,xxxx...",  // 可选；声音克隆参考音频
  "prompt_wav_base64": "data:audio/wav;base64,yyyy...",       // 可选；极致克隆模式参考音频
  "prompt_text": "参考音频对应的文本",                          // 可选；仅随 prompt_wav 使用（极致克隆）
  "control_instruction": "年轻女性，温柔甜美",                  // 可选；声音设计 / 可控克隆风格描述（别名 instruct）
  "model_id": "openbmb/VoxCPM2",
  "device": "auto",                     // auto / cpu / mps / cuda / cuda:N
  "cfg_value": 2.0,                     // 可选；CFG 引导强度（克隆建议 2.0-2.5）
  "inference_timesteps": 10,            // 可选；LocDiT 流匹配步数（质量 20-30 更佳）
  "denoise": true,                      // 可选；是否对参考音频降噪（需加载 denoiser）
  "normalize": false,                   // 可选；文本规范化（配音建议 true）
  "optimize": false,                    // 可选；torch.compile 优化（仅 CUDA）
  "min_len": 2,                         // 可选；最小音频长度
  "max_len": 4096,                      // 可选；最大 token 长度
  "retry_badcase": true,                // 可选；检测 badcase 自动重试（仅音文比）
  "retry_badcase_max_times": 3,         // 可选；badcase 最大重试次数
  "retry_badcase_ratio_threshold": 6.0, // 可选；音文比阈值
  "trim_silence_vad": true,             // 可选；克隆时对参考音频做 VAD 静音裁剪（改善对齐）
  "quality_retry": true,                // 可选；检测静音/削顶/过低音量后换 seed 重试
  "quality_retry_max": 2,               // 可选；质量重试最大次数
  "seed": 123456789,                    // 可选；固定 seed（每角色固定以保证跨片段一致）
  "session_id": "char_01_seg_1",        // 可选；配音会话 ID，同一角色跨片段复用以保持音色连续
  "session_continue": true,             // 可选；续写模式（首片段可不传或 false，自动建会话）
  "target_duration_ms": 2200,           // 可选；目标输出时长（配音对齐原片时长）
  "duration_tolerance_ms": 100,         // 可选；时长容差，超容差触发保调 time-stretch
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
  "quality_issues": [],                 // [] 表示无检测到问题
  "quality_retried": false,             // 是否触发了换 seed 质量重试
  "prompt_cache_hit": false,            // 参考音频 prompt 缓存是否命中
  "session": {"used": true, "created": true, "accumulated_text_len": 24}, // 配音会话状态
  "duration_alignment": {               // 时长对齐信息
    "target_duration_ms": 2200,
    "stretch": {"applied": true, "actual_duration": 2.5, "target_duration": 2.2, "rate": 1.136},
    "actual_duration_seconds": 2.2
  },
  "generation_params": {
    "cfg_value": 2.0,
    "inference_timesteps": 10,
    "denoise": true,
    "normalize": false,
    "min_len": 2,
    "max_len": 4096,
    "retry_badcase": true,
    "retry_badcase_max_times": 3,
    "retry_badcase_ratio_threshold": 6.0,
    "trim_silence_vad": true,
    "quality_retry": true,
    "quality_retry_max": 2,
    "session_id": "char_01_seg_1",
    "session_continue": true
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

#!/usr/bin/env python3
"""Wyoming protocol server for Kokoro TTS.

Wraps kokoro-onnx in a Wyoming-compatible TCP server for use with
Home Assistant and other Wyoming clients.
"""
import argparse
import asyncio
import hashlib
import logging
import signal
import sys
import time
from collections import OrderedDict
from functools import partial
from typing import Optional

import kokoro_onnx.config
from wyoming.error import Error
from wyoming.server import AsyncEventHandler
from kokoro_onnx import Kokoro
from kokoro_onnx.log import log
import numpy as np

from wyoming.info import Attribution, TtsProgram, TtsVoice, TtsVoiceSpeaker, Describe, Info
from wyoming.server import AsyncServer
from wyoming.tts import Synthesize
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
import re

_LOGGER = log.getChild(__name__)
VERSION = "0.3.0"

# Maximum text length to prevent resource exhaustion on extremely long inputs.
_MAX_TEXT_LENGTH = 5000

# Maximum number of cached synthesis results to keep in memory.
_MAX_CACHE_ENTRIES = 128


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    text = ' '.join(text.strip().split())
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def clean_text(text: str) -> str:
    """Strip markup artifacts that LLMs sometimes include in responses.

    Removes markdown bold/italic markers, heading markers, and other
    formatting that shouldn't be spoken aloud.
    """
    # Strip markdown bold/italic: **text** or __text__ or *text* or _text_
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    # Strip markdown headings: ## Heading
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Strip bullet points: - item or * item
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    # Collapse multiple spaces/newlines
    text = ' '.join(text.split())
    return text.strip()


def get_model_voices(model: Kokoro) -> list[TtsVoice]:
    return [
        TtsVoice(
            name=voice_id,
            description=voice_id,
            attribution=Attribution(
                name="", url=""
            ),
            installed=True,
            version=None,
            languages=[
                "en" if voice_id.startswith("a") else
                "it" if voice_id.startswith("i") else
                "jp" if voice_id.startswith('j') else
                "cn" if voice_id.startswith('z') else
                "es" if voice_id.startswith('e') else
                "fr" if voice_id.startswith('f') else
                "hi" if voice_id.startswith("h") else "en"
            ],
            speakers=[
                TtsVoiceSpeaker(name=voice_id.split("_")[1])
            ]
        )
        for voice_id in model.voices.keys()
    ]


class SynthesisCache:
    """LRU cache for synthesis results, keyed on (text, voice, speed)."""

    def __init__(self, max_entries: int = _MAX_CACHE_ENTRIES):
        self._cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self._max = max_entries
        self._hits = 0
        self._misses = 0

    def _key(self, text: str, voice: str, speed: float) -> str:
        return hashlib.md5(
            f"{text}|{voice}|{speed}".encode()
        ).hexdigest()

    def get(self, text: str, voice: str, speed: float
            ) -> Optional[tuple[np.ndarray, int]]:
        k = self._key(text, voice, speed)
        if k in self._cache:
            self._cache.move_to_end(k)
            self._hits += 1
            return self._cache[k]
        self._misses += 1
        return None

    def put(self, text: str, voice: str, speed: float,
            audio: np.ndarray, sr: int) -> None:
        k = self._key(text, voice, speed)
        self._cache[k] = (audio, sr)
        self._cache.move_to_end(k)
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> str:
        total = self._hits + self._misses
        rate = (self._hits / total * 100) if total else 0
        return f"{self._hits}/{total} hits ({rate:.0f}%)"


class KokoroEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, kokoro_instance,
                 default_voice: str, default_speed: float,
                 synth_semaphore: asyncio.Semaphore,
                 cache: SynthesisCache,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kokoro = kokoro_instance
        self.default_voice = default_voice
        self.default_speed = default_speed
        self.wyoming_info_event = wyoming_info.event()
        self._semaphore = synth_semaphore
        self._cache = cache

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming protocol events."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if not Synthesize.is_type(event.type):
            _LOGGER.warning("Unexpected event: %s", event)
            return True

        try:
            return await self._handle_synthesize(event)
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_synthesize(self, event: Event) -> Optional[bool]:
        try:
            synthesize = Synthesize.from_event(event)
            t_start = time.monotonic()

            voice_name = self.default_voice
            if synthesize.voice and synthesize.voice.name:
                voice_name = synthesize.voice.name

            # Clean markup from LLM output and validate
            text = clean_text(synthesize.text)

            if not text:
                _LOGGER.warning("Empty text after cleaning, skipping synthesis")
                await self.write_event(AudioStart(
                    rate=kokoro_onnx.config.SAMPLE_RATE, width=2, channels=1
                ).event())
                await self.write_event(AudioStop().event())
                return True

            if len(text) > _MAX_TEXT_LENGTH:
                _LOGGER.warning(
                    "Text truncated from %d to %d chars", len(text), _MAX_TEXT_LENGTH
                )
                text = text[:_MAX_TEXT_LENGTH]

            # Check cache for exact match (common for repeated phrases)
            cached = self._cache.get(text, voice_name, self.default_speed)
            if cached is not None:
                audio, sr = cached
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                await self.write_event(AudioStart(
                    rate=sr, width=2, channels=1
                ).event())
                await self.write_event(AudioChunk(
                    audio=audio_bytes, rate=sr, width=2, channels=1
                ).event())
                await self.write_event(AudioStop().event())
                t_done = time.monotonic()
                _LOGGER.info(
                    "Cache hit: voice=%s, %.0fms, text=\"%s\"",
                    voice_name, (t_done - t_start) * 1000, text[:80]
                )
                return True

            sentences = split_into_sentences(text)

            # Serialize synthesis to prevent concurrent ONNX inference
            # from competing for CPU and degrading latency for all clients.
            async with self._semaphore:
                i = 0
                t_bytes = 0
                t_first_audio = None
                all_audio = []

                for sentence in sentences:
                    stream = self.kokoro.create_stream(
                        sentence,
                        voice=voice_name,
                        speed=self.default_speed,
                        lang="en-us" if voice_name.startswith("a") else "en-gb"
                    )

                    if i == 0:
                        await self.write_event(
                            AudioStart(
                                rate=kokoro_onnx.config.SAMPLE_RATE,
                                width=2,
                                channels=1,
                            ).event()
                        )
                        i += 1

                    async for audio, sample_rate in stream:
                        if t_first_audio is None:
                            t_first_audio = time.monotonic()
                        all_audio.append(audio)
                        audio_int16 = (audio * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                        t_bytes += len(audio_bytes)

                        await self.write_event(
                            AudioChunk(
                                audio=audio_bytes,
                                rate=kokoro_onnx.config.SAMPLE_RATE,
                                width=2,
                                channels=1,
                            ).event()
                        )

            await self.write_event(AudioStop().event())

            # Cache the result for future requests
            if all_audio:
                combined = np.concatenate(all_audio)
                self._cache.put(
                    text, voice_name, self.default_speed,
                    combined, kokoro_onnx.config.SAMPLE_RATE
                )

            t_done = time.monotonic()
            first_ms = (t_first_audio - t_start) * 1000 if t_first_audio else 0
            total_ms = (t_done - t_start) * 1000
            audio_dur = t_bytes / (kokoro_onnx.config.SAMPLE_RATE * 2)
            _LOGGER.info(
                "Synthesized: voice=%s, first_audio=%.0fms, total=%.0fms, "
                "audio=%.1fs, %d bytes, cache=%s, text=\"%s\"",
                voice_name, first_ms, total_ms, audio_dur, t_bytes,
                self._cache.stats, synthesize.text[:80],
            )

            return True

        except Exception as e:
            _LOGGER.exception("Error synthesizing: %s", e)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Wyoming protocol server for Kokoro TTS"
    )
    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10210",
        help="Server URI (default: tcp://0.0.0.0:10210)"
    )
    parser.add_argument(
        "--voice",
        default="af_heart",
        help="Default voice ID (default: af_heart). "
             "Use --list-voices to see available voices."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--model",
        default="kokoro-v1.0.onnx",
        help="Path to ONNX model file (default: kokoro-v1.0.onnx)"
    )
    parser.add_argument(
        "--voices",
        default="voices-v1.0.bin",
        help="Path to voices file (default: voices-v1.0.bin)"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup synthesis at startup"
    )
    parser.add_argument(
        "--max-cache",
        type=int,
        default=_MAX_CACHE_ENTRIES,
        help=f"Max cached synthesis results (default: {_MAX_CACHE_ENTRIES})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.debug:
        log.setLevel(level=logging.DEBUG)

    _LOGGER.info("Loading model: %s", args.model)
    kokoro_instance = Kokoro(args.model, args.voices)

    if args.list_voices:
        for voice_id in sorted(kokoro_instance.voices.keys()):
            print(voice_id)
        return

    if args.voice not in kokoro_instance.voices:
        _LOGGER.error(
            "Voice '%s' not found. Available: %s",
            args.voice,
            ", ".join(sorted(kokoro_instance.voices.keys()))
        )
        sys.exit(1)

    if not args.no_warmup:
        _LOGGER.info("Warming up with voice '%s'...", args.voice)
        t_warmup = time.monotonic()
        kokoro_instance.create("warmup", voice=args.voice, speed=args.speed)
        _LOGGER.info(
            "Warmup complete in %.0fms",
            (time.monotonic() - t_warmup) * 1000
        )

    # Shared state across all client handlers
    synth_semaphore = asyncio.Semaphore(1)
    cache = SynthesisCache(max_entries=args.max_cache)

    wyoming_voices = get_model_voices(kokoro_instance)
    wyoming_info = Info(
        tts=[TtsProgram(
            name="Kokoro",
            description="Kokoro TTS via Wyoming protocol",
            attribution=Attribution(
                name="Kokoro TTS",
                url="https://huggingface.co/hexgrad/Kokoro-82M",
            ),
            installed=True,
            voices=sorted(wyoming_voices, key=lambda v: v.name),
            version=VERSION,
        )]
    )

    _LOGGER.info(
        "Starting on %s (voice=%s, speed=%.1f, cache=%d)",
        args.uri, args.voice, args.speed, args.max_cache
    )
    server = AsyncServer.from_uri(args.uri)

    try:
        loop = asyncio.get_event_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                s, lambda: asyncio.create_task(server.stop())
            )
    except (NotImplementedError, OSError):
        _LOGGER.debug("Signal handlers not available (container environment)")

    await server.run(
        partial(
            KokoroEventHandler,
            wyoming_info,
            kokoro_instance,
            args.voice,
            args.speed,
            synth_semaphore,
            cache,
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

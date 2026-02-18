# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import json
import time
from collections.abc import AsyncGenerator
from http import HTTPStatus
from uuid import uuid4

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from vllm import envs
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.openai.realtime.protocol import (
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    SessionCreated,
    TranscriptionDelta,
    TranscriptionDone,
)
from vllm.entrypoints.openai.realtime.serving import OpenAIServingRealtime
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

logger = init_logger(__name__)


class RealtimeConnection:
    """Manages WebSocket lifecycle and state for realtime transcription.

    This class handles:
    - WebSocket connection lifecycle (accept, receive, send, close)
    - Event routing (session.update, append, commit)
    - Audio buffering via asyncio.Queue
    - Generation task management with automatic pipeline rotation
    - Silence-loop recovery (non-text streak detection)
    - Error handling and cleanup

    Long-running streaming sessions are kept healthy by two mechanisms:

    1. **Proactive rotation** — after ``MAX_SESSION_TOKENS`` tokens the
       generation pipeline is cleanly restarted, bounding CPU-side
       metadata growth (token lists, multimodal features, block tables)
       that would otherwise grow without limit.

    2. **Deaf recovery** — if the model emits ``MAX_NON_TEXT_STREAK``
       consecutive non-text tokens (silence/pad/control, token id < 1000)
       the pipeline is aborted and restarted.  This matches the strategy
       used by `voxtral.c <https://github.com/antirez/voxtral.c>`_
       (``STREAM_MAX_NON_TEXT_STREAK = 64``).

    Audio continues flowing from the WebSocket into ``audio_queue``
    throughout any restart, so no frames are lost.
    """

    # Tokens below this id are non-text (silence, pad, control).
    # voxtral.c uses ``token < 1000``.
    NON_TEXT_TOKEN_THRESHOLD: int = 1000

    # Abort after this many consecutive non-text tokens.
    MAX_NON_TEXT_STREAK: int = 64

    # Give up after this many consecutive restarts that produce zero text.
    MAX_EMPTY_RESTARTS: int = 5

    # Proactively restart the pipeline after this many output tokens to
    # bound unbounded CPU-side metadata growth in the scheduler.
    MAX_SESSION_TOKENS: int = 2048

    def __init__(self, websocket: WebSocket, serving: OpenAIServingRealtime):
        self.websocket = websocket
        self.connection_id = f"ws-{uuid4()}"
        self.serving = serving
        self.audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self.generation_task: asyncio.Task | None = None

        self._is_connected = False
        self._is_model_validated = False

        self._max_audio_filesize_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB

    async def handle_connection(self):
        """Main connection loop."""
        await self.websocket.accept()
        logger.debug("WebSocket connection accepted: %s", self.connection_id)
        self._is_connected = True

        # Send session created event
        await self.send(SessionCreated())

        try:
            while True:
                message = await self.websocket.receive()
                # Handle disconnect (receive() doesn't raise
                # WebSocketDisconnect like receive_text() does)
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(code=message.get("code", 1000))
                if message.get("bytes"):
                    # Binary frame: raw PCM16 audio, no JSON/base64
                    audio_bytes = message["bytes"]
                    audio_array = (
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    if len(audio_array) > 0:
                        self.audio_queue.put_nowait(audio_array)
                    continue
                text = message.get("text")
                if text is None:
                    continue
                try:
                    event = json.loads(text)
                    await self.handle_event(event)
                except json.JSONDecodeError:
                    await self.send_error("Invalid JSON", "invalid_json")
                except Exception as e:
                    logger.exception("Error handling event: %s", e)
                    await self.send_error(str(e), "processing_error")
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s", self.connection_id)
            self._is_connected = False
        except Exception as e:
            logger.exception("Unexpected error in connection: %s", e)
        finally:
            await self.cleanup()

    def _check_model(self, model: str | None) -> None | ErrorResponse:
        if self.serving._is_model_supported(model):
            return None

        return self.serving.create_error_response(
            message=f"The model `{model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    async def handle_event(self, event: dict):
        """Route events to handlers.

        Supported event types:
        - session.update: Configure model
        - input_audio_buffer.append: Add audio chunk to queue
        - input_audio_buffer.commit: Start transcription generation
        """
        event_type = event.get("type")
        if event_type == "session.update":
            logger.debug("Session updated: %s", event)
            self._check_model(event["model"])
            self._is_model_validated = True
        elif event_type == "input_audio_buffer.append":
            append_event = InputAudioBufferAppend(**event)
            try:
                audio_bytes = base64.b64decode(append_event.audio)
                # Convert PCM16 bytes to float32 numpy array
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                if len(audio_array) / 1024**2 > self._max_audio_filesize_mb:
                    raise VLLMValidationError(
                        "Maximum file size exceeded",
                        parameter="audio_filesize_mb",
                        value=len(audio_array) / 1024**2,
                    )
                if len(audio_array) == 0:
                    raise VLLMValidationError("Can't process empty audio.")

                # Put audio chunk in queue
                self.audio_queue.put_nowait(audio_array)

            except Exception as e:
                logger.error("Failed to decode audio: %s", e)
                await self.send_error("Invalid audio data", "invalid_audio")

        elif event_type == "input_audio_buffer.commit":
            if not self._is_model_validated:
                err_msg = (
                    "Model not validated. Make sure to validate the"
                    " model by sending a session.update event."
                )
                await self.send_error(
                    err_msg,
                    "model_not_validated",
                )

            commit_event = InputAudioBufferCommit(**event)
            # final signals that the audio is finished
            if commit_event.final:
                self.audio_queue.put_nowait(None)
            else:
                await self.start_generation()
        else:
            await self.send_error(f"Unknown event type: {event_type}", "unknown_event")

    async def audio_stream_generator(self) -> AsyncGenerator[np.ndarray, None]:
        """Generator that yields audio chunks from the queue."""
        while True:
            audio_chunk = await self.audio_queue.get()
            if audio_chunk is None:  # Sentinel value to stop
                break
            yield audio_chunk

    async def start_generation(self):
        """Start the transcription generation task."""
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        self.generation_task = asyncio.create_task(self._generation_loop())

    async def _generation_loop(self):
        """Run generation with automatic restart on rotation or deafness.

        Each iteration creates a fresh audio/token pipeline and runs
        ``_run_generation`` until it returns.  The return value indicates
        whether to restart (``True``/``False``) or stop (``None``).
        """
        empty_restarts = 0
        restart_count = 0

        while self._is_connected:
            audio_stream = self.audio_stream_generator()
            input_stream = asyncio.Queue[list[int]]()
            streaming_input_gen = self.serving.transcribe_realtime(
                audio_stream, input_stream
            )

            try:
                produced_text = await self._run_generation(
                    streaming_input_gen, input_stream
                )
            except Exception:
                logger.exception(
                    "%s: _run_generation raised unexpectedly",
                    self.connection_id,
                )
                produced_text = None

            if produced_text is not None:
                # Restart path — push a sentinel so the old
                # audio_stream_generator exits naturally (calling
                # aclose() on a running async generator raises
                # RuntimeError).
                self.audio_queue.put_nowait(None)
                await asyncio.sleep(0.1)

            if produced_text is None:
                break

            if not produced_text:
                empty_restarts += 1
                logger.warning(
                    "%s: restart produced no text (empty_restarts=%d/%d)",
                    self.connection_id,
                    empty_restarts,
                    self.MAX_EMPTY_RESTARTS,
                )
                if empty_restarts >= self.MAX_EMPTY_RESTARTS:
                    logger.error(
                        "%s: too many empty restarts, giving up",
                        self.connection_id,
                    )
                    break
            else:
                empty_restarts = 0

            restart_count += 1
            logger.info(
                "%s: restarting generation pipeline (restart #%d, queue=%d)",
                self.connection_id,
                restart_count,
                self.audio_queue.qsize(),
            )

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ) -> bool | None:
        """Run a single generation session.

        Returns:
            ``True``  — restart requested, text was produced.
            ``False`` — restart requested, no text was produced.
            ``None``  — normal completion or error, do not restart.
        """
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        full_text = ""
        deaf_restart = False

        prompt_token_ids_len: int = 0
        completion_tokens_len: int = 0

        non_text_streak: int = 0
        total_tokens: int = 0
        total_text_tokens: int = 0
        last_log = time.monotonic()

        try:
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=1,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            async for output in result_gen:
                if output.outputs and len(output.outputs) > 0:
                    if not prompt_token_ids_len and output.prompt_token_ids:
                        prompt_token_ids_len = len(output.prompt_token_ids)

                    delta = output.outputs[0].text
                    token_ids = output.outputs[0].token_ids
                    full_text += delta
                    total_tokens += 1

                    input_stream.put_nowait(list(token_ids))
                    await self.send(TranscriptionDelta(delta=delta))

                    completion_tokens_len += len(token_ids)

                    # --- Deaf detection ---
                    if token_ids and token_ids[-1] < self.NON_TEXT_TOKEN_THRESHOLD:
                        non_text_streak += 1
                    else:
                        non_text_streak = 0
                        total_text_tokens += 1

                    if non_text_streak >= self.MAX_NON_TEXT_STREAK:
                        logger.warning(
                            "%s: deaf detected — %d consecutive non-text "
                            "tokens (last_tid=%s, total=%d, text=%d). "
                            "Aborting for restart.",
                            request_id[:20],
                            non_text_streak,
                            token_ids,
                            total_tokens,
                            total_text_tokens,
                        )
                        deaf_restart = True
                        await self.serving.engine_client.abort(request_id)
                        break

                    # --- Proactive rotation ---
                    if total_tokens >= self.MAX_SESSION_TOKENS:
                        logger.info(
                            "%s: proactive rotation at %d tokens "
                            "(%d text). Restarting.",
                            request_id[:20],
                            total_tokens,
                            total_text_tokens,
                        )
                        await self.serving.engine_client.abort(request_id)
                        return True

                    # --- Periodic status log ---
                    now = time.monotonic()
                    if now - last_log >= 10.0:
                        logger.info(
                            "%s: %d tok, %d text, streak=%d, queue=%d",
                            request_id[:20],
                            total_tokens,
                            total_text_tokens,
                            non_text_streak,
                            self.audio_queue.qsize(),
                        )
                        last_log = now

                if not self._is_connected:
                    break

            if deaf_restart:
                return total_text_tokens > 0

            # Normal completion
            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )
            await self.send(TranscriptionDone(text=full_text, usage=usage))

            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

            return None

        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
            return None

    async def send(
        self, event: SessionCreated | TranscriptionDelta | TranscriptionDone
    ):
        """Send event to client."""
        data = event.model_dump_json()
        await self.websocket.send_text(data)

    async def send_error(self, message: str, code: str | None = None):
        """Send error event to client."""
        error_event = ErrorEvent(error=message, code=code)
        await self.websocket.send_text(error_event.model_dump_json())

    async def cleanup(self):
        """Cleanup resources."""
        # Signal audio stream to stop
        self.audio_queue.put_nowait(None)

        # Cancel generation task if running
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()

        logger.debug("Connection cleanup complete: %s", self.connection_id)

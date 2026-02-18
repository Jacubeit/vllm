# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import json
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
    - Generation task management
    - Deaf recovery (non-text streak detection with automatic restart)
    - Error handling and cleanup
    """

    # Deaf recovery constants (modeled after voxtral.c).
    # Voxtral tokens below this threshold are non-text (silence, pad,
    # control tokens). voxtral.c uses token < 1000.
    NON_TEXT_TOKEN_THRESHOLD: int = 1000
    # After this many consecutive non-text tokens, abort and restart
    # the generation session to recover from silence loops.
    # voxtral.c uses STREAM_MAX_NON_TEXT_STREAK = 64.
    MAX_NON_TEXT_STREAK: int = 64
    # Maximum consecutive restarts with no text produced before giving
    # up (mirrors voxtral.c's STREAM_EMPTY_RESTARTS_FOR_FULL_RESET).
    MAX_EMPTY_RESTARTS: int = 5

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
                    # Binary frame = raw PCM16, skip JSON/base64
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

        # Start generation loop (handles deaf recovery restarts internally)
        self.generation_task = asyncio.create_task(self._generation_loop())

    async def _generation_loop(self):
        """Run generation with automatic deaf recovery restarts.

        When the model enters a silence loop (consecutive non-text tokens
        exceeding MAX_NON_TEXT_STREAK), the current engine request is aborted
        and a fresh generation pipeline is started. Audio continues flowing
        from the WebSocket into audio_queue throughout.

        This mirrors voxtral.c's decoder restart strategy:
        - STREAM_MAX_NON_TEXT_STREAK = 64 triggers restart
        - STREAM_EMPTY_RESTARTS_FOR_FULL_RESET = 2 escalates
        """
        empty_restarts = 0

        while self._is_connected:
            # Create a fresh audio + token pipeline for each attempt.
            audio_stream = self.audio_stream_generator()
            input_stream = asyncio.Queue[list[int]]()
            streaming_input_gen = self.serving.transcribe_realtime(
                audio_stream, input_stream
            )

            try:
                produced_text = await self._run_generation(
                    streaming_input_gen, input_stream
                )
            finally:
                # Explicitly close generators so buffer_realtime_audio's
                # finally block runs immediately (cancels feed_audio and
                # feed_tokens tasks), releasing the audio_stream_generator
                # before we create a new one in the next iteration.
                await streaming_input_gen.aclose()
                await audio_stream.aclose()

            if produced_text is None:
                # Normal completion or disconnect — stop the loop.
                break

            if not produced_text:
                empty_restarts += 1
                logger.warning(
                    "%s: deaf restart #%d produced no text (empty_restarts=%d/%d)",
                    self.connection_id,
                    empty_restarts,
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
                # Successful text production — reset the counter.
                empty_restarts = 0

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ) -> bool | None:
        """Run the generation and stream results back to the client.

        This method:
        1. Creates sampling parameters from session config
        2. Passes the streaming input generator to engine.generate()
        3. Streams transcription.delta events as text is generated
        4. Detects deaf episodes (consecutive non-text tokens) and aborts
        5. Sends final transcription.done event with usage stats
        6. Feeds generated token IDs back to input_stream for next iteration

        Returns:
            True if text was produced before stopping.
            False if deaf recovery triggered with no text produced.
            None if the generation completed normally or disconnected
            (caller should NOT restart).
        """
        import time as _time

        request_id = f"rt-{self.connection_id}-{uuid4()}"
        full_text = ""
        deaf_restart = False

        prompt_token_ids_len: int = 0
        completion_tokens_len: int = 0

        # Deaf recovery state
        non_text_streak: int = 0
        total_tokens: int = 0
        total_text_tokens: int = 0
        last_log = _time.monotonic()

        try:
            # Create sampling params
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=1,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

            # Pass the streaming input generator to the engine
            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            # Stream results back to client as they're generated
            async for output in result_gen:
                if output.outputs and len(output.outputs) > 0:
                    if not prompt_token_ids_len and output.prompt_token_ids:
                        prompt_token_ids_len = len(output.prompt_token_ids)

                    delta = output.outputs[0].text
                    token_ids = output.outputs[0].token_ids
                    full_text += delta
                    total_tokens += 1

                    # append output to input
                    input_stream.put_nowait(list(token_ids))
                    await self.send(TranscriptionDelta(delta=delta))

                    completion_tokens_len += len(token_ids)

                    # --- Deaf detection ---
                    # Check if the last token is non-text (silence/pad/control).
                    # voxtral.c: token < 1000 is non-text.
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
                        # Abort the engine request to free resources.
                        await self.serving.engine_client.abort(request_id)
                        break

                    # --- Periodic status log ---
                    now = _time.monotonic()
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
                # Return whether any text was produced during this run.
                return total_text_tokens > 0

            # Normal completion — send done event.
            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )
            await self.send(TranscriptionDone(text=full_text, usage=usage))

            # Clear queue for next utterance
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

            return None  # Normal completion — don't restart.

        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
            return None  # Error — don't restart.

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

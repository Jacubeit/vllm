# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
from typing_extensions import deprecated

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        reasoning_ended: bool | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        if self.structured_output_request is not None:
            self.structured_output_request.reasoning_ended = reasoning_ended
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Used in forced preemption (reset_prefix_cache) with async scheduling.
        self.discard_latest_async_tokens = False

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        self.update_block_hashes()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # Used for streaming
        self.resumable = resumable
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

        # Session compaction: cumulative tokens trimmed from the front.
        # Used for RoPE position continuity and sliding-window eviction.
        self._num_tokens_compacted: int = 0

    @property
    @deprecated(
        "Request.eos_token_id will be removed in v0.18. "
        "Please use Request.sampling_params.eos_token_id instead."
    )
    def eos_token_id(self) -> int | None:
        if self.sampling_params is None:
            return None

        return self.sampling_params.eos_token_id

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            reasoning_ended=request.reasoning_ended,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def num_compactable_mm_features(self) -> int:
        """Count mm_features fully within the already-computed range.

        A feature is compactable when its entire token span lies before
        ``num_computed_tokens`` — meaning the encoder output has been
        consumed and folded into the decoder KV cache.

        Returns:
            Number of leading mm_features that can be safely trimmed.
        """
        num_computed = self.num_computed_tokens
        count = 0
        for feat in self.mm_features:
            end_pos = feat.mm_position.offset + feat.mm_position.length
            if end_pos <= num_computed:
                count += 1
            else:
                # mm_features are ordered by offset — once we hit one that
                # isn't fully computed, the rest won't be either.
                break
        return count

    def compact_mm_features(self, trim_count: int) -> None:
        """Trim *trim_count* leading mm_features in-place.

        The caller is responsible for freeing encoder-cache entries for
        the trimmed features **before** calling this method (because
        ``encoder_cache_manager.free_encoder_input`` looks up features
        by index).

        Also clears ``block_hashes`` / ``_block_hasher`` so that
        ``update_block_hashes`` and ``cache_blocks`` become no-ops.
        The sliding-window manager already evicts old blocks on the GPU
        side, and ``cache_blocks`` is guarded for empty hashes.

        Does **not** touch ``_all_token_ids``, ``prompt_token_ids``,
        ``num_computed_tokens``, or ``num_prompt_tokens`` — the model
        runner depends on these staying in sync with the KV cache.
        """
        if trim_count <= 0:
            return

        del self.mm_features[:trim_count]

        # Clear block hashes — the scheduler doesn't need them for
        # streaming requests (prefix caching is irrelevant mid-stream),
        # and clearing prevents the O(n) growth.  cache_blocks() is
        # guarded for empty hashes so this is safe.
        self.block_hashes.clear()
        self._block_hasher = None

    def compact_session(
        self, keep_tokens: int, block_size: int
    ) -> tuple[int, int, int]:
        """Compact CPU-side session metadata while preserving KV cache.

        Trims tokens, mm_features, and block_hashes from the front,
        keeping the most recent tokens.  The KV cache entries for kept
        tokens remain valid — no re-prefill is needed.  The caller must
        trim the leading blocks from the KV block table to match.

        ``num_computed_tokens`` is reduced by ``trim_count`` so it stays
        a valid physical index into the (now shorter) token lists.
        ``_num_tokens_compacted`` tracks the cumulative logical offset
        for RoPE position continuity and sliding-window eviction.

        Must be called only when ``_output_token_ids`` is empty (i.e.
        after output tokens have been folded into the prompt during a
        streaming session update).

        The caller must:
        1. Free encoder-cache entries for trimmed mm_features **before**
           this call (``free_encoder_input`` looks up features by index).
        2. Trim leading KV cache blocks **after** this call
           (``compact_blocks(request_id, num_trimmed_blocks)``).

        Args:
            keep_tokens: Number of recent tokens to retain.
            block_size: KV cache block size for alignment.

        Returns:
            (tokens_trimmed, mm_features_trimmed, blocks_trimmed)
        """
        num_tokens = len(self._all_token_ids)
        if num_tokens <= keep_tokens:
            return 0, 0, 0

        assert not self._output_token_ids, (
            "compact_session must be called after output tokens are "
            "folded into prompt_token_ids"
        )

        trim_count = num_tokens - keep_tokens
        # Align to block boundary so block trimming is clean.
        trim_count = (trim_count // block_size) * block_size
        if trim_count <= 0:
            return 0, 0, 0

        # --- Count mm_features to trim ---
        # Features whose entire span falls within the trimmed range.
        mm_trim = 0
        for feat in self.mm_features:
            end_pos = feat.mm_position.offset + feat.mm_position.length
            if end_pos <= trim_count:
                mm_trim += 1
            else:
                break

        # Clamp trim_count so we never trim into the first remaining
        # mm_feature.  Re-align to block boundary after clamping.
        if mm_trim < len(self.mm_features):
            first_kept = self.mm_features[mm_trim]
            trim_count = min(trim_count, first_kept.mm_position.offset)
            trim_count = (trim_count // block_size) * block_size

        if trim_count == 0:
            return 0, 0, 0

        # --- Trim token lists from the front ---
        del self._all_token_ids[:trim_count]
        assert self.prompt_token_ids is not None
        del self.prompt_token_ids[:trim_count]

        # --- Trim mm_features and rebase remaining offsets ---
        if mm_trim > 0:
            del self.mm_features[:mm_trim]
        for i, feat in enumerate(self.mm_features):
            new_offset = feat.mm_position.offset - trim_count
            assert new_offset >= 0, (
                f"mm_feature[{i}] offset {feat.mm_position.offset} "
                f"< trim_count {trim_count}"
            )
            self.mm_features[i] = replace(
                feat,
                mm_position=replace(feat.mm_position, offset=new_offset),
            )

        # --- Adjust counters (keep KV, no re-prefill) ---
        self.num_computed_tokens -= trim_count
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._num_tokens_compacted += trim_count

        # --- Clear output token ids (should already be empty) ---
        self._output_token_ids.clear()

        # --- Clear prefix-cache state ---
        self.num_cached_tokens = -1
        self.skip_reading_prefix_cache = True

        # --- Clear block hashes (chain is broken) ---
        self.block_hashes.clear()
        self._block_hasher = None

        num_trimmed_blocks = trim_count // block_size
        return trim_count, mm_trim, num_trimmed_blocks

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds()

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
}

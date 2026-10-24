"""Word alignment adapted from OpenAI Whisper (MIT).

Native attention stays on ComfyUI's selected backend. The alignment pass captures
only the calibrated cross-attention heads; DTW runs on CPU without Triton/Numba.
"""
from __future__ import annotations

import base64
import gzip
import itertools
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from comfy.model_management import throw_exception_if_processing_interrupted

TOKENS_PER_SECOND = 50

_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
    "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    "large": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    "large-v3-turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
    "turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
}

def alignment_heads(model_name, dims):
    mask = np.frombuffer(gzip.decompress(base64.b85decode(_ALIGNMENT_HEADS[model_name])), dtype=bool)
    mask = mask.reshape(dims.n_text_layer, dims.n_text_head)
    return {layer: np.flatnonzero(row).tolist() for layer, row in enumerate(mask) if row.any()}

def median_filter(x, filter_width):
    if x.shape[-1] <= filter_width // 2:
        return x
    padded = F.pad(x, (filter_width // 2, filter_width // 2), mode="reflect")
    return padded.unfold(-1, filter_width, 1).sort()[0][..., filter_width // 2]

def backtrace(trace: np.ndarray):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))

        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")

    result = np.array(result)
    return result[::-1, :].T

def dtw_cpu(x: np.ndarray):
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, M + 1):
        if j % 100 == 0:
            throw_exception_if_processing_interrupted()
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return backtrace(trace)

def dtw(x):
    return dtw_cpu(x.double().cpu().numpy())

@dataclass
class WordTiming:
    word: str
    tokens: List[int]
    start: float
    end: float
    probability: float

def find_alignment(
    model: "Whisper",
    tokenizer: Tokenizer,
    text_tokens: List[int],
    mel: torch.Tensor,
    num_frames: int,
    *,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
) -> List[WordTiming]:
    if len(text_tokens) == 0:
        return []

    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ]
    ).to(model.device)

    scores = []
    logits = model(mel.unsqueeze(0), tokens.unsqueeze(0), alignment_scores=scores)[0]
    sampled_logits = logits[len(tokenizer.sot_sequence) :, : tokenizer.eot]
    token_probs = sampled_logits.softmax(dim=-1)
    text_token_probs = token_probs[torch.arange(len(text_tokens), device=tokens.device), tokens[len(tokenizer.sot_sequence) + 1:-1]].tolist()
    weights = torch.cat(scores, dim=0)
    weights = weights[:, :, : num_frames // 2]
    weights = (weights * qk_scale).softmax(dim=-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std
    weights = median_filter(weights, medfilt_width)

    matrix = weights.mean(axis=0)
    matrix = matrix[len(tokenizer.sot_sequence) : -1]
    text_indices, time_indices = dtw(-matrix)

    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    if len(word_tokens) <= 1:
        # return on eot only
        # >>> np.pad([], (1, 0))
        # array([0.])
        # This results in crashes when we lookup jump_times with float, like
        # IndexError: arrays used as indices must be of integer (or boolean) type
        return []
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    word_probabilities = [
        np.mean(text_token_probs[i:j])
        for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
    ]

    return [
        WordTiming(word, tokens, start, end, probability)
        for word, tokens, start, end, probability in zip(
            words, word_tokens, start_times, end_times, word_probabilities
        )
    ]

def merge_punctuations(alignment: List[WordTiming], prepended: str, appended: str):
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous.word.startswith(" ") and previous.word.strip() in prepended:
            # prepend it to the following word
            following.word = previous.word + following.word
            following.tokens = previous.tokens + following.tokens
            previous.word = ""
            previous.tokens = []
        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous.word.endswith(" ") and following.word in appended:
            # append it to the previous word
            previous.word = previous.word + following.word
            previous.tokens = previous.tokens + following.tokens
            following.word = ""
            following.tokens = []
        else:
            i = j
        j += 1

def add_word_timestamps(
    *,
    segments: List[dict],
    model: "Whisper",
    tokenizer: Tokenizer,
    mel: torch.Tensor,
    num_frames: int,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    last_speech_timestamp: float,
    time_offset: float,
    **kwargs,
):
    if len(segments) == 0:
        return

    text_tokens_per_segment = [
        [token for token in segment["tokens"] if token < tokenizer.eot]
        for segment in segments
    ]

    text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
    alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames, **kwargs)
    word_durations = np.array([t.end - t.start for t in alignment])
    word_durations = word_durations[word_durations.nonzero()]
    median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
    median_duration = min(0.7, float(median_duration))
    max_duration = median_duration * 2

    # hack: truncate long words at sentence boundaries.
    # a better segmentation algorithm based on VAD should be able to replace this.
    if len(word_durations) > 0:
        sentence_end_marks = ".。!！?？"
        # ensure words at sentence boundaries are not longer than twice the median word duration.
        for i in range(1, len(alignment)):
            if alignment[i].end - alignment[i].start > max_duration:
                if alignment[i].word in sentence_end_marks:
                    alignment[i].end = alignment[i].start + max_duration
                elif alignment[i - 1].word in sentence_end_marks:
                    alignment[i].start = alignment[i].end - max_duration

    merge_punctuations(alignment, prepend_punctuations, append_punctuations)

    word_index = 0

    for segment, text_tokens in zip(segments, text_tokens_per_segment):
        saved_tokens = 0
        words = []

        while word_index < len(alignment) and saved_tokens < len(text_tokens):
            timing = alignment[word_index]

            if timing.word:
                words.append(
                    dict(
                        word=timing.word,
                        start=round(time_offset + timing.start, 2),
                        end=round(time_offset + timing.end, 2),
                        probability=timing.probability,
                    )
                )

            saved_tokens += len(timing.tokens)
            word_index += 1

        # hack: truncate long words at segment boundaries.
        # a better segmentation algorithm based on VAD should be able to replace this.
        if len(words) > 0:
            # ensure the first and second word after a pause is not longer than
            # twice the median word duration.
            if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                words[0]["end"] - words[0]["start"] > max_duration
                or (
                    len(words) > 1
                    and words[1]["end"] - words[0]["start"] > max_duration * 2
                )
            ):
                if (
                    len(words) > 1
                    and words[1]["end"] - words[1]["start"] > max_duration
                ):
                    boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
                    words[0]["end"] = words[1]["start"] = boundary
                words[0]["start"] = max(0, words[0]["end"] - max_duration)

            # prefer the segment-level start timestamp if the first word is too long.
            if (
                segment["start"] < words[0]["end"]
                and segment["start"] - 0.5 > words[0]["start"]
            ):
                words[0]["start"] = max(
                    0, min(words[0]["end"] - median_duration, segment["start"])
                )
            else:
                segment["start"] = words[0]["start"]

            # prefer the segment-level end timestamp if the last word is too long.
            if (
                segment["end"] > words[-1]["start"]
                and segment["end"] + 0.5 < words[-1]["end"]
            ):
                words[-1]["end"] = max(
                    words[-1]["start"] + median_duration, segment["end"]
                )
            else:
                segment["end"] = words[-1]["end"]

            last_speech_timestamp = segment["end"]

        segment["words"] = words

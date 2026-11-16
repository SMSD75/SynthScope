"""Shared support for VLM detectors: a released vision-language model that answers "is this image real or fake?".

One class serves every such detector; what differs per detector is a ``VLMConfig`` (the registry spec's ``vlm``): the
transformers model class, the official prompt, the decoding, the answer marker and the verdict words, the official
verdict parser, and an optional pixel cap.

Score (one image at a time):

    PIL image -> the model's own processor (official resize; ``max_pixels`` caps it only when the config says so)
              -> chat request (system prompt + [image, user prompt], the model's chat template)
              -> ``generate`` (greedy by default: deterministic; ``decoding="official"`` = the authors' sampling)
    verdict step = the first generated token after ``answer_marker`` that is a verdict word ("Ġfake" / "Ġreal", ...)
    logit        = l(fake) - l(real) at that step, for the emitted token and its same-spelling counterpart
                   (e.g. "Ġreal" -> "Ġfake"); the raw logits of the LM head (before any logits processor)
    P(fake)      = sigmoid(logit)
    verdict      = the official parser on the generated text ("fake" / "real", None if unparsed)

The raw logits are read with a forward hook on the LM head, which runs once per generated token, so only the verdict
columns are kept (``output_logits`` would hold the full vocabulary row of every step). Under greedy decoding the
emitted token is the argmax, so the sign of the logit agrees with the emitted verdict token.

If the answer marker is generated but no verdict word follows, the logit is read at the step right after the marker
for the first pair's space-prefixed spelling (``score_source="after_marker"``); if the marker never appears (e.g. the
generation hit ``max_new_tokens``), P(fake) = 0.5, logit 0 (``score_source="none"``).

Weights: a Hugging Face model folder (``config.json``, ``*.safetensors``, processor and tokenizer files) under
``$SYNTHSCOPE_CKPT_DIR/<checkpoint>``, loaded with ``from_pretrained`` into the official transformers class; any
missing, unexpected or mismatched tensor raises (``check_loading_info``). Inference only: there is no training path.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import transformers
from PIL import Image


@dataclass
class VLMConfig:
    model_class: str                                  # transformers class, e.g. "Qwen3VLForConditionalGeneration"
    system_prompt: str                                # official system prompt ("" = none)
    user_prompt: str                                  # official user text, after the image
    parser: Callable[[str], Optional[str]]            # official verdict parser: text -> "fake" / "real" / None
    answer_marker: str                                # the verdict is the first verdict word generated after it
    verdict_pairs: tuple = (("fake", "real"),)        # (fake word, real word) with the same spelling
    decoding: dict = field(default_factory=dict)      # generate kwargs of the default (greedy) score
    official_decoding: dict = field(default_factory=dict)   # generate kwargs of the authors' setting (sampled)
    chat_template_kwargs: dict = field(default_factory=dict)
    processor: Optional[str] = None                   # processor folder (relative to the weights folder); None = it
    max_pixels: Optional[int] = None                  # image pixel cap; None = the processor's default
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    hf_repo: str = ""                                 # official weights (for the download hint)
    hf_revision: str = ""


def check_loading_info(info: dict, where: str) -> None:
    """Raise when ``from_pretrained(..., output_loading_info=True)`` reports any missing, unexpected or mismatched
    tensor or an error: the released weights must fill the official class exactly."""
    bad = {k: sorted(str(x) for x in v) for k, v in info.items() if v}
    if bad:
        detail = "; ".join(f"{k}={v[:8]}" + (f" (+{len(v) - 8})" if len(v) > 8 else "") for k, v in bad.items())
        raise RuntimeError(f"{where}: checkpoint mismatch vs the official model class: {detail}")


def verdict_pair_map(tokenizer, pairs) -> dict:
    """{token id: (fake id, real id)} for every single token that spells a verdict word (any prefix, e.g. "Ġreal",
    "real") and whose same-spelling counterpart is also one token ("Ġreal" <-> "Ġfake")."""
    words = {w for pair in pairs for w in pair}
    vocab = tokenizer.get_vocab()
    out = {}
    for tok, tid in vocab.items():
        if tokenizer.convert_tokens_to_string([tok]).strip() not in words:
            continue
        for fake, real in pairs:
            if fake in tok and tok.replace(fake, real) in vocab:
                out[tid] = (tid, vocab[tok.replace(fake, real)])
                break
            if real in tok and tok.replace(real, fake) in vocab:
                out[tid] = (vocab[tok.replace(real, fake)], tid)
                break
    return out


def default_pair(tokenizer, pairs) -> Optional[tuple]:
    """(fake id, real id) of the first pair spelled with a leading space, as after "<answer>"; None if not one token."""
    fake, real = pairs[0]
    f = tokenizer.encode(" " + fake, add_special_tokens=False)
    r = tokenizer.encode(" " + real, add_special_tokens=False)
    return (f[0], r[0]) if len(f) == 1 and len(r) == 1 else None


def marker_end(pieces: list, marker: str) -> Optional[int]:
    """Index of the first generated token after the one that completes ``marker`` in the concatenated ``pieces``
    (the per-token text), or None."""
    text = ""
    for t, piece in enumerate(pieces):
        text += piece
        if marker in text:
            return t + 1
    return None


def locate_verdict(gen: list, pieces: list, marker: str, pair_map: dict) -> tuple:
    """(marker end, verdict step): the first step at or after the marker end whose token is in ``pair_map``."""
    start = marker_end(pieces, marker)
    if start is None:
        return None, None
    for t in range(start, len(gen)):
        if gen[t] in pair_map:
            return start, t
    return start, None


def verdict_score(gen: list, pieces: list, rows: torch.Tensor, column: dict, marker: str, pair_map: dict,
                  fallback_pair: Optional[tuple]) -> dict:
    """The score fields from the generated ids ``gen``, their text ``pieces`` and the recorded verdict logits ``rows``
    ([steps, len(column)], row t = the logits that produced ``gen[t]``; ``column``: token id -> column)."""
    start, step = locate_verdict(gen, pieces, marker, pair_map)
    if step is not None:
        pair, source = pair_map[gen[step]], "verdict_token"
    elif start is not None and start < len(gen) and fallback_pair is not None:
        step, pair, source = start, fallback_pair, "after_marker"
    else:
        return {"logit": 0.0, "fake_probability": 0.5, "score_source": "none", "verdict_step": None,
                "verdict_token": None, "pair_ids": None}
    row = rows[step].float()
    gap = float(row[column[pair[0]]] - row[column[pair[1]]])
    return {"logit": gap, "fake_probability": float(torch.sigmoid(torch.tensor(gap, dtype=torch.float64))),
            "score_source": source, "verdict_step": step,
            "verdict_token": gen[step] if source == "verdict_token" else None, "pair_ids": pair}


class LogitRecorder:
    """Context manager: a forward hook on the LM head that keeps the last position's logits at ``ids`` for every
    forward (one per generated token in ``generate``), as float32 like ``output_logits``."""

    def __init__(self, head: torch.nn.Module, ids: list):
        self.head, self.rows, self.handle = head, [], None
        self.index = torch.tensor(ids, dtype=torch.long)

    def _hook(self, module, inputs, output):
        self.rows.append(output[:, -1, self.index.to(output.device)].to(torch.float32))

    def __enter__(self):
        self.rows = []
        self.handle = self.head.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()

    def stacked(self) -> torch.Tensor:
        return torch.cat(self.rows).cpu() if self.rows else torch.empty(0, len(self.index))


class SynthScopeVLM(torch.nn.Module):
    """A released VLM detector loaded from its Hugging Face folder, scored as described in the module docstring."""

    def __init__(self, path, cfg: VLMConfig, device: str = "cuda"):
        super().__init__()
        path = Path(path)
        self.cfg, self.path = cfg, path
        model_cls = getattr(transformers, cfg.model_class)
        self.hf, info = model_cls.from_pretrained(
            str(path), dtype=getattr(torch, cfg.dtype), attn_implementation=cfg.attn_implementation,
            device_map=device, output_loading_info=True)
        check_loading_info(info, f"{path} ({cfg.model_class})")
        self.hf.eval()
        processor_dir = path / cfg.processor if cfg.processor else path
        self.processor = transformers.AutoProcessor.from_pretrained(str(processor_dir))
        self.tokenizer = self.processor.tokenizer
        self.images_kwargs = {}
        if cfg.max_pixels:
            size = self.processor.image_processor.size
            self.images_kwargs = {"size": {"shortest_edge": size.shortest_edge, "longest_edge": cfg.max_pixels}}
        self.prompt = build_prompt(self.processor, cfg)
        self.pair_map = verdict_pair_map(self.tokenizer, cfg.verdict_pairs)
        self.fallback_pair = default_pair(self.tokenizer, cfg.verdict_pairs)
        ids = sorted({i for pair in self.pair_map.values() for i in pair} | set(self.fallback_pair or ()))
        self.record_ids, self.column = ids, {tid: c for c, tid in enumerate(ids)}
        eos = self.hf.generation_config.eos_token_id
        self.eos = {eos} if isinstance(eos, int) else set(eos or [])

    @property
    def device(self):
        return self.hf.device

    def inputs(self, image: Image.Image) -> dict:
        """The processor's model inputs for one image (official preprocessing; batch of one)."""
        return self.processor(text=[self.prompt], images=[image], return_tensors="pt",
                              **({"images_kwargs": self.images_kwargs} if self.images_kwargs else {})).to(self.device)

    @torch.inference_mode()
    def score(self, image: Image.Image, decoding: str = "greedy", seed: Optional[int] = None) -> dict:
        """Generate the answer for ``image`` and read the verdict logit (see the module docstring). ``decoding``:
        "greedy" (the config's ``decoding``) or "official" (``official_decoding``, seeded with ``seed``)."""
        kwargs = {"greedy": self.cfg.decoding, "official": self.cfg.official_decoding}[decoding]
        inputs = self.inputs(image)
        n_prompt = inputs["input_ids"].shape[1]
        if seed is not None:
            torch.manual_seed(seed)
        with LogitRecorder(self.hf.get_output_embeddings(), self.record_ids) as rec:
            seq = self.hf.generate(**inputs, **kwargs)
        gen = seq[0, n_prompt:].tolist()
        pieces = [self.tokenizer.decode([t]) for t in gen]
        out = verdict_score(gen, pieces, rec.stacked(), self.column, self.cfg.answer_marker, self.pair_map,
                            self.fallback_pair)
        response = self.tokenizer.decode(gen, skip_special_tokens=True)
        grid = inputs["image_grid_thw"][0].tolist()
        patch = self.processor.image_processor.patch_size
        out.update({
            "verdict": self.cfg.parser(response),
            "verdict_token": None if out["verdict_token"] is None else self.tokenizer.convert_ids_to_tokens(out["verdict_token"]),
            "pair_tokens": None if out["pair_ids"] is None else self.tokenizer.convert_ids_to_tokens(list(out["pair_ids"])),
            "response": response,
            "rationale": response.split(self.cfg.answer_marker)[0].strip(),
            "new_tokens": len(gen),
            "finished": bool(gen) and gen[-1] in self.eos,
            "prompt_tokens": int(n_prompt),
            "input_size": [int(grid[2] * patch), int(grid[1] * patch)],     # [width, height] the vision encoder saw
            "generated_ids": gen,
            "decoding": decoding,
        })
        out.pop("pair_ids")
        return out


def build_prompt(processor, cfg: VLMConfig) -> str:
    """The official chat request as text: system prompt, then the user turn [image, user prompt]."""
    messages = []
    if cfg.system_prompt:
        messages.append({"role": "system", "content": cfg.system_prompt})
    messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg.user_prompt}]})
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **cfg.chat_template_kwargs)

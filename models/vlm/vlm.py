"""Shared support for VLM detectors: a released vision-language model that answers "is this image real or fake?".

One loader / scorer (``SynthScopeVLM``) serves every such detector. What differs is split in two:

* ``VLMFamily``: how a model family is loaded and fed. The base class covers a Hugging Face processor that takes
  ``text=[prompt], images=[image]`` (LLaVA-style models); ``QwenVLFamily`` adds the Qwen-VL specifics (a pixel cap
  through the processor's ``size``, the input size from ``image_grid_thw``). A family decides:
    - ``load_model``: the folder loaded strictly into the official transformers class, or a LoRA adapter folder
      (``adapter_config.json``) kept unmerged on a LOCAL base folder (``base_model``; nothing is fetched from the Hub);
    - ``load_processor``: the processor folder plus attribute overrides;
    - ``verdict_sequences``: the fake / real answers as token sequences (configurable, ``VLMConfig.verdict_sequences``);
      the score is read at their first differing token;
    - ``prompt``: the chat template on (``chat_template=True``) or the raw prompt as the official code sends it;
    - ``inputs`` / ``input_size``: the processor call and what the vision encoder saw.
* ``VLMConfig`` (the registry spec's ``vlm``): one detector's official request and score rule: the family, model class,
  prompts, image loader (``models/vlm/image.py``), optional PIL pre-resize hook, decoding, answer marker and end tag,
  verdict words, verdict mode, raw or processed logits, the official parser.

Score (one image at a time; ``score`` holds a lock because the logit recorder hooks the shared LM head):

    image   = ``image_loader`` (e.g. vLLM's EXIF transpose + white alpha composite) -> optional ``preprocess`` hook
    "generate" mode (reasoning models):
        generate with ``decoding`` (greedy) or ``official_decoding`` (the authors' sampling)
        verdict step = the first generated verdict word ("Ġfake" / "Ġreal", ...) after ``answer_marker`` (from the
                       start if None) and before ``answer_end``
        logits       = "raw": the LM head's output of that step (a forward hook, = ``output_logits``), or
                       "processed": the scores after generate's logits processors (e.g. a repetition penalty) and
                       before the sampling warpers, i.e. what greedy decoding takes the argmax of
    "teacher_forced" mode (the verdict is the first answer token): one forward pass on prompt + ``teacher_forced_suffix``
        + the tokens the two verdict sequences share (e.g. a SentencePiece "▁" before "▁fake" / "▁real"), the logits
        of the next token, i.e. at their first differing token
    logit   = l(fake) - l(real): generate mode, the emitted verdict token and its same-spelling counterpart ("Ġreal" <->
              "Ġfake"; the first differing tokens of the verdict sequences count too); teacher-forced, the first
              differing tokens of the verdict sequences; P(fake) = sigmoid(logit)
    verdict = the official parser on the generated text (teacher-forced: the sign of the logit)

No verdict token (no marker generated, e.g. a truncated output, or no verdict word between the marker and the end tag):
P(fake) = 0.5, logit 0, ``score_source="unparsed"`` (ruling U1). Under greedy decoding with ``logits="processed"`` (or
raw logits without a logits processor) the emitted token is the argmax, so the sign of the logit is the emitted word.

Relative ``base_model`` / ``processor`` folders are resolved against the checkpoint root (``SynthScopeVLM(root=...)``;
the engine passes ``$SYNTHSCOPE_CKPT_DIR``). Batch size 1 only. Inference only: there is no training path.
"""
import importlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import transformers
from PIL import Image
from safetensors.torch import load_file
from transformers import LogitsProcessor, LogitsProcessorList

from models.vlm.image import vllm_rgb


class VLMFamily:
    """Loading and input building of a model family whose processor takes ``text=[prompt], images=[image]``."""

    def load_model(self, path: Path, cfg: "VLMConfig", device: str, root: Path):
        """The model on ``device``, strictly loaded: the folder ``path`` into ``cfg.model_class``; or, with
        ``cfg.base_model`` (a local folder, relative names under ``root``), the base into it and ``path`` as a LoRA
        adapter kept unmerged (peft)."""
        model_cls = getattr(transformers, cfg.model_class)
        kwargs = dict(dtype=getattr(torch, cfg.dtype), attn_implementation=cfg.attn_implementation, device_map=device,
                      local_files_only=True, output_loading_info=True)
        if cfg.base_model is None:
            if (path / "adapter_config.json").exists():
                raise RuntimeError(f"{path} is a LoRA adapter: set VLMConfig.base_model to a local base folder "
                                   "(transformers would otherwise fetch the base named in adapter_config.json)")
            model, info = model_cls.from_pretrained(str(path), **kwargs)
            check_loading_info(info, f"{path} ({cfg.model_class})")
            return model.eval()
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(f"{path}: no adapter_config.json, but VLMConfig.base_model is set (a LoRA detector)")
        base = resolve_local(root, cfg.base_model)
        model, info = model_cls.from_pretrained(str(base), **kwargs)
        check_loading_info(info, f"{base} ({cfg.model_class})")
        return load_lora_unmerged(model, path).eval()

    def load_processor(self, path: Path, cfg: "VLMConfig", root: Path):
        """``cfg.processor`` (relative names under ``root``), else the weights folder if it holds processor files,
        else (a LoRA adapter folder) the base folder."""
        if cfg.processor:
            folder = resolve_local(root, cfg.processor)
        elif cfg.base_model and not has_processor(path):
            folder = resolve_local(root, cfg.base_model)
        else:
            folder = path
        processor = transformers.AutoProcessor.from_pretrained(str(folder), local_files_only=True)
        for name, value in cfg.processor_overrides.items():
            setattr(processor, name, value)
        return processor

    def prompt(self, processor, cfg: "VLMConfig") -> str:
        """The official request as text: the chat template over [system, user: [image, text]], or the raw prompt."""
        if not cfg.chat_template:
            return cfg.user_prompt
        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": cfg.user_prompt}]})
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             **cfg.chat_template_kwargs)

    def inputs(self, processor, prompt: str, image: Image.Image, cfg: "VLMConfig", device) -> dict:
        return processor(text=[prompt], images=[image], return_tensors="pt", **cfg.processor_kwargs).to(device)

    def verdict_sequences(self, tokenizer, cfg: "VLMConfig") -> tuple:
        """(fake ids, real ids): ``cfg.verdict_sequences`` (token strings) if set, else the first verdict pair's words
        with a leading space, as the tokenizer encodes them (SentencePiece / Llama: ["▁", "▁fake"])."""
        if cfg.verdict_sequences:
            seqs = tuple(tokenizer.convert_tokens_to_ids(list(seq)) for seq in cfg.verdict_sequences)
            unk = getattr(tokenizer, "unk_token_id", None)
            if any(i is None or i == unk for seq in seqs for i in seq):
                raise ValueError(f"verdict_sequences {cfg.verdict_sequences} contain tokens outside the vocabulary")
            return seqs
        fake, real = cfg.verdict_pairs[0]
        return (tokenizer.encode(" " + fake, add_special_tokens=False),
                tokenizer.encode(" " + real, add_special_tokens=False))

    def extend_inputs(self, inputs: dict, ids: list) -> dict:
        """``inputs`` with the text tokens ``ids`` appended (teacher forcing a shared prefix)."""
        if not ids:
            return inputs
        n = inputs["input_ids"].shape[1]
        extra = torch.tensor([ids], dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
        out = dict(inputs)
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.dim() == 2 and value.shape == (1, n):
                fill = extra if key == "input_ids" else (torch.ones_like(extra) if key == "attention_mask"
                                                         else torch.zeros_like(extra))
                out[key] = torch.cat([value, fill.to(value.dtype)], dim=1)
        return out

    def input_size(self, processor, inputs) -> Optional[list]:
        """[width, height] the vision encoder saw: the processed pixel tensor's size ([B, 3, H, W]), else None."""
        pixels = inputs.get("pixel_values")
        return [int(pixels.shape[-1]), int(pixels.shape[-2])] if pixels is not None and pixels.dim() == 4 else None


class QwenVLFamily(VLMFamily):
    """Qwen2-VL / Qwen2.5-VL / Qwen3-VL: the processor resizes to multiples of patch x merge within [min, max] pixels
    (``cfg.max_pixels`` lowers the max) and flattens the patches; the grid is ``image_grid_thw``."""

    def inputs(self, processor, prompt, image, cfg, device):
        kwargs = dict(cfg.processor_kwargs)
        if cfg.max_pixels:
            size = processor.image_processor.size
            kwargs["images_kwargs"] = {"size": {"shortest_edge": size.shortest_edge, "longest_edge": cfg.max_pixels}}
        return processor(text=[prompt], images=[image], return_tensors="pt", **kwargs).to(device)

    def input_size(self, processor, inputs):
        grid = inputs["image_grid_thw"][0].tolist()
        patch = processor.image_processor.patch_size
        return [int(grid[2] * patch), int(grid[1] * patch)]


QWEN_VL = QwenVLFamily()


@dataclass
class VLMConfig:
    model_class: str                                  # transformers class, e.g. "Qwen3VLForConditionalGeneration"
    system_prompt: str                                # official system prompt ("" = none)
    user_prompt: str                                  # official user text (after the image); the raw prompt if
                                                      # chat_template is False
    parser: Callable[[str], Optional[str]]            # official verdict parser: text -> "fake" / "real" / None
    family: VLMFamily = field(default_factory=VLMFamily)
    answer_marker: Optional[str] = None               # the verdict is the first verdict word after it (None: from the start)
    answer_end: Optional[str] = None                  # ... and before this text (e.g. "</answer>")
    verdict_pairs: tuple = (("fake", "real"),)        # (fake word, real word) with the same spelling
    verdict_sequences: Optional[tuple] = None         # ((fake tokens...), (real tokens...)) as token strings; None =
                                                      # the family's (the first pair with a leading space, encoded)
    verdict_mode: str = "generate"                    # "generate" | "teacher_forced" (one forward pass)
    teacher_forced_suffix: str = ""                   # appended to the prompt in teacher_forced mode
    logits: str = "raw"                               # "raw" LM-head logits | "processed" (after logits processors)
    decoding: dict = field(default_factory=dict)      # generate kwargs of the default (greedy) score
    official_decoding: dict = field(default_factory=dict)   # generate kwargs of the authors' setting (sampled)
    chat_template: bool = True
    chat_template_kwargs: dict = field(default_factory=dict)
    image_loader: Callable[[Image.Image], Image.Image] = vllm_rgb   # decoded file -> RGB, as the official server
    preprocess: Optional[Callable[[Image.Image], Image.Image]] = None   # PIL pre-resize hook (e.g. fetch_image)
    processor: Optional[str] = None                   # processor folder (relative to CHECKPOINTS_DIR); None = the
                                                      # weights folder, or the base folder for a LoRA adapter
    processor_overrides: dict = field(default_factory=dict)   # attributes set on the loaded processor
    processor_kwargs: dict = field(default_factory=dict)      # extra kwargs of the processor call
    base_model: Optional[str] = None                  # LoRA: the local base folder (relative to CHECKPOINTS_DIR)
    max_pixels: Optional[int] = None                  # image pixel cap; None = the processor's default
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    hf_repo: str = ""                                 # official weights (for the download hint)
    hf_revision: str = ""


def has_processor(folder: Path) -> bool:
    return any((Path(folder) / f).exists() for f in ("preprocessor_config.json", "processor_config.json"))


def resolve_local(root: Path, name: str) -> Path:
    """``name`` as a local folder (absolute, or relative to ``root``); raises instead of falling back to the Hub."""
    path = Path(name) if Path(name).is_absolute() else Path(root) / name
    if not any((path / f).exists() for f in ("config.json", "preprocessor_config.json", "processor_config.json")):
        raise FileNotFoundError(f"no local model / processor folder at {path} (nothing is downloaded)")
    return path


def check_loading_info(info: dict, where: str) -> None:
    """Raise when ``from_pretrained(..., output_loading_info=True)`` reports any missing, unexpected or mismatched
    tensor or an error: the released weights must fill the official class exactly."""
    bad = {k: sorted(str(x) for x in v) for k, v in info.items() if v}
    if bad:
        detail = "; ".join(f"{k}={v[:8]}" + (f" (+{len(v) - 8})" if len(v) > 8 else "") for k, v in bad.items())
        raise RuntimeError(f"{where}: checkpoint mismatch vs the official model class: {detail}")


def load_lora_unmerged(model, adapter: Path):
    """``model`` with the LoRA adapter folder ``adapter`` (``adapter_config.json`` + ``adapter_model.safetensors``)
    attached unmerged (merging in bf16 changes greedy outputs). Strict: the adapter tensors the model holds afterwards
    must be exactly the file's, key for key. Needs peft, an optional dependency imported only here."""
    try:
        peft = importlib.import_module("peft")
    except ImportError as e:
        raise ImportError("a LoRA VLM detector needs peft (pip install peft)") from e
    model = peft.PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    saved = load_file(str(adapter / "adapter_model.safetensors"))
    loaded = peft.get_peft_model_state_dict(model)
    missing, unexpected = sorted(set(saved) - set(loaded)), sorted(set(loaded) - set(saved))
    differ = sorted(k for k in set(saved) & set(loaded) if not torch.equal(saved[k].to(loaded[k]), loaded[k]))
    if missing or unexpected or differ:
        raise RuntimeError(f"{adapter}: LoRA adapter mismatch: not loaded={missing[:8]} not in the file={unexpected[:8]} "
                           f"different={differ[:8]}")
    return model


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


def divergence(fake: list, real: list) -> int:
    """Index of the first token where the two verdict sequences differ (the shared prefix is teacher-forced)."""
    k = next((i for i, (f, r) in enumerate(zip(fake, real)) if f != r), None)
    if k is None:
        raise ValueError(f"verdict sequences {fake} / {real}: one is a prefix of the other, no token tells them apart")
    return k


def token_pieces(tokenizer, ids: list) -> list:
    """The text each generated token adds, from decoding the growing prefix: ``decode(ids[:t+1])`` minus ``decode(ids[:t])``.
    Decoding a token alone drops its leading space on SentencePiece / Llama tokenizers ("▁is" -> "is"), so a
    multi-word marker such as "This is a" would never be found; the prefix decode keeps it. A trailing U+FFFD (an
    incomplete UTF-8 character) is re-read once its remaining bytes arrive. Identical to the per-token decode for
    byte-level BPE tokenizers (Qwen) wherever a token is a whole character."""
    pieces, prev = [], ""
    for t in range(len(ids)):
        cur = tokenizer.decode(ids[:t + 1])
        base = prev if cur.startswith(prev) else prev.rstrip("\ufffd")
        pieces.append(cur[len(base):] if cur.startswith(base) else "")
        prev = cur
    return pieces


def marker_end(pieces: list, marker: Optional[str]) -> Optional[int]:
    """Index of the first generated token after the one that completes ``marker`` in the concatenated ``pieces``
    (the per-token text); 0 without a marker; None if the marker is never generated."""
    if not marker:
        return 0
    text = ""
    for t, piece in enumerate(pieces):
        text += piece
        if marker in text:
            return t + 1
    return None


def locate_verdict(gen: list, pieces: list, marker: Optional[str], pair_map: dict, end: Optional[str] = None) -> tuple:
    """(marker end, verdict step): the first step at or after the marker end whose token is in ``pair_map``, searched
    until the text since the marker contains ``end``."""
    start = marker_end(pieces, marker)
    if start is None:
        return None, None
    text = ""
    for t in range(start, len(gen)):
        if gen[t] in pair_map:
            return start, t
        text += pieces[t]
        if end and end in text:
            break
    return start, None


def gap_fields(gap: float) -> dict:
    return {"logit": gap, "fake_probability": float(torch.sigmoid(torch.tensor(gap, dtype=torch.float64)))}


def verdict_score(gen: list, pieces: list, rows: torch.Tensor, column: dict, marker: Optional[str], pair_map: dict,
                  end: Optional[str] = None) -> dict:
    """The score fields from the generated ids ``gen``, their text ``pieces`` and the recorded verdict logits ``rows``
    ([steps, len(column)], row t = the logits that produced ``gen[t]``; ``column``: token id -> column)."""
    _, step = locate_verdict(gen, pieces, marker, pair_map, end)
    if step is None:
        return {"logit": 0.0, "fake_probability": 0.5, "score_source": "unparsed", "verdict_step": None,
                "verdict_token": None, "pair_ids": None}
    pair = pair_map[gen[step]]
    row = rows[step].float()
    return {**gap_fields(float(row[column[pair[0]]] - row[column[pair[1]]])), "score_source": "verdict_token",
            "verdict_step": step, "verdict_token": gen[step], "pair_ids": pair}


class LogitRecorder:
    """Context manager: a forward hook on the LM head that keeps the last position's logits at ``ids`` for every
    forward (one per generated token in ``generate``), as float32 like ``output_logits`` (the raw logits)."""

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


class ProcessedLogitRecorder(LogitsProcessor):
    """A logits processor that keeps the scores at ``ids`` as generate's own processors left them (custom processors
    run after those and before the sampling warpers) and returns them unchanged."""

    def __init__(self, ids: list):
        self.rows = []
        self.index = torch.tensor(ids, dtype=torch.long)

    def __call__(self, input_ids, scores):
        self.rows.append(scores[:, self.index.to(scores.device)].to(torch.float32))
        return scores

    def stacked(self) -> torch.Tensor:
        return torch.cat(self.rows).cpu() if self.rows else torch.empty(0, len(self.index))


class SynthScopeVLM(torch.nn.Module):
    """A released VLM detector loaded from its local folder(s), scored as described in the module docstring."""

    def __init__(self, path, cfg: VLMConfig, device: str = "cuda", root=None):
        super().__init__()
        path = Path(path)
        root = Path(root) if root is not None else path.parent      # where relative base / processor names live
        self.cfg, self.path, self.family = cfg, path, cfg.family
        self.hf = self.family.load_model(path, cfg, device, root)
        self.processor = self.family.load_processor(path, cfg, root)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.prompt = self.family.prompt(self.processor, cfg)
        self.pair_map = verdict_pair_map(self.tokenizer, cfg.verdict_pairs)
        self.sequences = self.family.verdict_sequences(self.tokenizer, cfg)
        self.split = divergence(*self.sequences)
        self.pair = (self.sequences[0][self.split], self.sequences[1][self.split])   # first differing tokens
        for tid in self.pair:                         # e.g. a word that only exists as a subtoken
            self.pair_map.setdefault(tid, self.pair)
        ids = sorted({i for pair in self.pair_map.values() for i in pair})
        self.record_ids, self.column = ids, {tid: c for c, tid in enumerate(ids)}
        eos = self.hf.generation_config.eos_token_id
        self.eos = {eos} if isinstance(eos, int) else set(eos or [])
        self._lock = threading.Lock()                 # one score at a time: the recorder hooks the shared LM head

    @property
    def device(self):
        return self.hf.device

    def image(self, image: Image.Image) -> Image.Image:
        """The image as the official pipeline hands it to the processor: ``image_loader``, then ``preprocess``."""
        image = self.cfg.image_loader(image)
        return self.cfg.preprocess(image) if self.cfg.preprocess else image

    def inputs(self, image: Image.Image, suffix: str = "") -> dict:
        """The processor's model inputs for one already-loaded image (batch of one)."""
        return self.family.inputs(self.processor, self.prompt + suffix, image, self.cfg, self.device)

    @torch.inference_mode()
    def score(self, image: Image.Image, decoding: str = "greedy", seed: Optional[int] = None) -> dict:
        """Score ``image`` (a decoded file; see ``image``). ``decoding``: "greedy" (the config's ``decoding``) or
        "official" (``official_decoding``, seeded with ``seed``); not used in teacher_forced mode."""
        with self._lock:
            image = self.image(image)
            if self.cfg.verdict_mode == "teacher_forced":
                return self._single_pass(image)
            kwargs = {"greedy": self.cfg.decoding, "official": self.cfg.official_decoding}[decoding]
            inputs = self.inputs(image)
            n_prompt = inputs["input_ids"].shape[1]
            if seed is not None:
                torch.manual_seed(seed)
            if self.cfg.logits == "processed":
                rec = ProcessedLogitRecorder(self.record_ids)
                seq = self.hf.generate(**inputs, **kwargs, logits_processor=LogitsProcessorList([rec]))
            else:
                with LogitRecorder(self.hf.get_output_embeddings(), self.record_ids) as rec:
                    seq = self.hf.generate(**inputs, **kwargs)
            gen = seq[0, n_prompt:].tolist()
            pieces = token_pieces(self.tokenizer, gen)
            out = verdict_score(gen, pieces, rec.stacked(), self.column, self.cfg.answer_marker, self.pair_map,
                                self.cfg.answer_end)
            response = self.tokenizer.decode(gen, skip_special_tokens=True)
            rationale = response.split(self.cfg.answer_marker)[0].strip() if self.cfg.answer_marker else response
            out.update({
                "verdict": self.cfg.parser(response),
                "verdict_token": None if out["verdict_token"] is None else
                self.tokenizer.convert_ids_to_tokens(out["verdict_token"]),
                "pair_tokens": None if out["pair_ids"] is None else
                self.tokenizer.convert_ids_to_tokens(list(out["pair_ids"])),
                "response": response, "rationale": rationale, "new_tokens": len(gen),
                "finished": bool(gen) and gen[-1] in self.eos, "generated_ids": gen,
            })
            out.pop("pair_ids")
            return self._common(out, inputs, n_prompt, decoding)

    def _single_pass(self, image: Image.Image) -> dict:
        """teacher_forced mode: prompt + suffix + the verdict sequences' shared prefix, then the next-token logits of
        their first differing tokens."""
        prefix = list(self.sequences[0][:self.split])
        inputs = self.family.extend_inputs(self.inputs(image, self.cfg.teacher_forced_suffix), prefix)
        row = self.hf(**inputs, logits_to_keep=1).logits[0, -1].float()
        out = gap_fields(float(row[self.pair[0]] - row[self.pair[1]]))
        out.update({"score_source": "teacher_forced", "verdict_step": None,
                    "verdict": "fake" if out["logit"] > 0 else "real", "verdict_token": None,
                    "pair_tokens": self.tokenizer.convert_ids_to_tokens(list(self.pair)),
                    "forced_prefix": self.tokenizer.convert_ids_to_tokens(prefix),
                    "response": "", "rationale": "", "new_tokens": 0, "finished": True, "generated_ids": []})
        return self._common(out, inputs, inputs["input_ids"].shape[1], "teacher_forced")

    def _common(self, out: dict, inputs: dict, n_prompt: int, decoding: str) -> dict:
        out.update({"prompt_tokens": int(n_prompt), "input_size": self.family.input_size(self.processor, inputs),
                    "decoding": decoding})
        return out

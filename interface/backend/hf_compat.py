"""Version-tolerant parameter names for checkpoints that hold a Hugging Face ``transformers`` backbone.

A checkpoint that stores a whole ``transformers`` module (PGC's DINOv2-L, ...) carries the parameter names of the
``transformers`` release it was saved with. A later release may rename a module, and then a strict load fails on the
renamed tensors (``missing=[...q_proj...] unexpected=[...attention.query...]``) although the weights are identical.

``translate_hf_keys`` maps such a state dict onto the model's names, in either direction (an older checkpoint into a
newer model, or a newer checkpoint into an older model), and only when the rename is exact: every renamed key must land
on a model key that the state dict does not already have, with the same shape. Anything else is returned untouched, so
a real mismatch still reaches the loader's strict check.
"""
from __future__ import annotations

# (older name, newer name) of one parameter-path fragment. Every rule below was derived by building the model under
# both releases and matching the keys one-to-one by shape (transformers 5.5.4 and 5.22.2 are the two that were compared).
HF_RENAMES: tuple[tuple[str, str], ...] = (
    # Dinov2Model (DINOv2 / PGC): attention.attention.{query,key,value} and attention.output.dense (transformers
    # <= 5.5.4) became attention.{q,k,v,o}_proj (5.22.2); 192 keys for DINOv2-L, all other keys unchanged.
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
)


def _rename(key: str, rules: tuple[tuple[str, str], ...]) -> str:
    """Apply the first rule whose fragment occurs in ``key`` (the fragments are disjoint, so at most one does)."""
    for old, new in rules:
        if old in key:
            return key.replace(old, new, 1)
    return key


def translate_hf_keys(sd: dict, model_state: dict) -> tuple[dict, str]:
    """Rename the ``transformers`` parameter names of ``sd`` that ``model_state`` does not know, when a rule of
    ``HF_RENAMES`` maps them exactly onto it.

    ``model_state`` is the model's ``state_dict()``. Returns ``(state_dict, note)``: ``note`` says which direction was
    applied and how many keys moved (empty when ``sd`` is returned unchanged). A direction is applied only if every key
    it changes lands on a model key that is not already in ``sd`` (no collisions) and has the same shape.
    """
    unknown = [k for k in sd if k not in model_state]
    if not unknown:
        return sd, ""
    for label, rules in (
        ("older -> newer", HF_RENAMES),
        ("newer -> older", tuple((new, old) for old, new in HF_RENAMES)),
    ):
        moved = {k: _rename(k, rules) for k in unknown}
        moved = {k: new for k, new in moved.items() if new != k}
        if not moved:
            continue
        targets = list(moved.values())
        exact = (
            len(set(targets)) == len(targets)
            and all(new in model_state and new not in sd for new in targets)
            and all(
                hasattr(sd[k], "shape") and tuple(sd[k].shape) == tuple(model_state[new].shape)
                for k, new in moved.items()
            )
        )
        if exact:
            return {moved.get(k, k): v for k, v in sd.items()}, f"{label}: renamed {len(moved)} key(s)"
    return sd, ""


__all__ = ["HF_RENAMES", "translate_hf_keys"]

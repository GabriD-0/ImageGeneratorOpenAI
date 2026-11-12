from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Linha com apenas "---negative---" (aceita também "negativo", qualquer caixa).
_NEGATIVE_MARKER = re.compile(
    r"^[ \t]*-{3,}[ \t]*negativ[eo][ \t]*-{3,}[ \t]*$", re.IGNORECASE | re.MULTILINE
)


@dataclass(frozen=True)
class Prompt:
    positive: str
    negative: str = ""


def _squash(text: str) -> str:
    """Junta linhas e remove espaços repetidos (o prompt costuma vir quebrado em ~80 colunas)."""
    return " ".join(text.split())


def load_prompt(path: Path) -> Prompt:
    if not path.is_file():
        raise FileNotFoundError(f"arquivo de prompt não encontrado: {path}")
    raw = path.read_text(encoding="utf-8")
    parts = _NEGATIVE_MARKER.split(raw, maxsplit=1)
    positive = _squash(parts[0])
    negative = _squash(parts[1]) if len(parts) > 1 else ""
    if not positive:
        raise ValueError(f"o prompt positivo está vazio em {path}")
    return Prompt(positive=positive, negative=negative)


def build_prompt(prompt: Prompt, negative: str = "append") -> str:
    """Texto final enviado à API."""
    if negative not in ("append", "ignore"):
        raise ValueError(f"modo de negative inválido: {negative!r}")
    if negative == "append" and prompt.negative:
        return f"{prompt.positive} Avoid: {prompt.negative}."
    return prompt.positive

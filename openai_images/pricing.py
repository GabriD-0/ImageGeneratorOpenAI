from __future__ import annotations

import math
import re
from dataclasses import dataclass

# ----------------------------------------------------------------------------- limites de tamanho livre
# Valem para gpt-image-2, 2.5 e 2.6 (calculadora do guia e referência da Images API).
MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400
MAX_EDGE = 3840
MAX_RATIO = 3
MULTIPLE = 16
EXPERIMENTAL_PIXELS = 2560 * 1440   # a referência chama de "experimental" o que passa de 2560x1440

# Tamanhos aceitos por todos os modelos (e os únicos dos modelos antigos).
FIXED_SIZES = ("1024x1024", "1024x1536", "1536x1024")

# ----------------------------------------------------------------------------- preços (US$ por 1 milhão de tokens)


@dataclass(frozen=True)
class Rates:
    text_in: float
    image_in: float
    image_out: float
    text_out: float = 0.0   # só o 1.5 cobra saída de texto; nos demais não é cobrada


# Batch API: metade do preço padrão (página de preços: US$ 4 / 15 por 1M de tokens de imagem no 2.5, contra 8 / 30).
BATCH_DISCOUNT = 0.5

_RATES_NEW = Rates(text_in=5.0, image_in=8.0, image_out=30.0)      # gpt-image-2, 2.5 e 2.6
_RATES_15 = Rates(text_in=5.0, image_in=8.0, image_out=32.0, text_out=10.0)
_RATES_MINI = Rates(text_in=2.0, image_in=2.5, image_out=8.0)

# Tokens-base por qualidade na calculadora oficial.
_BASE_25 = {"low": 16, "medium": 24, "high": 48, "xhigh": 64, "max": 96}
_BASE_2 = {"low": 16, "medium": 48, "high": 96}
# O 2.6 usa a mesma base do 2.5, multiplicada nas qualidades baixas (alta, xhigh e max: x1).
_MULT_26 = {"low": 4.2, "medium": 3.3, "high": 1.0, "xhigh": 1.0, "max": 1.0}

# Tabela oficial de US$ por imagem (saída) dos modelos de tamanho fixo, do guia de imagens.
_LEGACY_USD = {
    "gpt-image-1.5": {
        "low": (0.009, 0.013, 0.013), "medium": (0.034, 0.05, 0.05), "high": (0.133, 0.2, 0.2)},
    "gpt-image-1-mini": {
        "low": (0.005, 0.006, 0.006), "medium": (0.011, 0.015, 0.015), "high": (0.036, 0.052, 0.052)},
}


@dataclass(frozen=True)
class Model:
    name: str
    rates: Rates
    qualities: tuple[str, ...]
    custom_sizes: bool      # aceita WIDTHxHEIGHT livre
    batch: bool             # tem preço de Batch API (50% do padrão)
    in_reference: bool      # consta na referência da Images API (o 2.6 só aparece no guia/modelos/preços)
    note: str = ""


MODELS: dict[str, Model] = {m.name: m for m in (
    Model("gpt-image-2.5-flare", _RATES_NEW, tuple(_BASE_25), True, True, True,
          "rápido, boa qualidade para o dia a dia"),
    Model("gpt-image-2.5-sunburst", _RATES_NEW, tuple(_BASE_25), True, True, True,
          "o mais capaz; melhor para edição precisa"),
    Model("gpt-image-2", _RATES_NEW, tuple(_BASE_2), True, True, True,
          "geração anterior; só low, medium e high"),
    Model("gpt-image-2.6-flare", _RATES_NEW, tuple(_BASE_25), True, False, False,
          "novo; low/medium gastam mais tokens que no 2.5; não consta na referência da Images API"),
    Model("gpt-image-2.6-sunburst", _RATES_NEW, tuple(_BASE_25), True, False, False,
          "novo; low/medium gastam mais tokens que no 2.5; não consta na referência da Images API"),
    Model("gpt-image-1.5", _RATES_15, tuple(_LEGACY_USD["gpt-image-1.5"]), False, True, True,
          "tamanhos fixos; nenhum passa de 1536 px"),
    Model("gpt-image-1-mini", _RATES_MINI, tuple(_LEGACY_USD["gpt-image-1-mini"]), False, True, True,
          "o mais barato, tamanhos fixos; nenhum passa de 1536 px"),
)}

_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def canonical(model: str) -> str:
    """``gpt-image-2.5-flare-2026-09-08`` -> ``gpt-image-2.5-flare``."""
    return _SNAPSHOT_SUFFIX.sub("", model)


def get_model(model: str) -> Model:
    try:
        return MODELS[canonical(model)]
    except KeyError:
        raise ValueError(f"modelo sem preço conhecido: {model!r}. Conhecidos: {', '.join(MODELS)}") from None


# ----------------------------------------------------------------------------- tamanho


def parse_size(size: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", size.strip().lower())
    if not match:
        raise ValueError(f"tamanho inválido: {size!r} (use LARGURAxALTURA, ex.: 1088x1632)")
    return int(match[1]), int(match[2])


def size_problems(width: int, height: int) -> list[str]:
    """Violações das regras de tamanho livre; lista vazia = válido."""
    problems = []
    if width % MULTIPLE or height % MULTIPLE:
        problems.append(f"largura e altura devem ser múltiplas de {MULTIPLE}")
    pixels = width * height
    if pixels > MAX_PIXELS:
        problems.append(f"no máximo {MAX_PIXELS:,} pixels")
    if pixels < MIN_PIXELS:
        problems.append(f"no mínimo {MIN_PIXELS:,} pixels")
    if max(width, height) > MAX_EDGE:
        problems.append(f"o lado maior não pode passar de {MAX_EDGE}px")
    if max(width, height) / min(width, height) > MAX_RATIO:
        problems.append("proporção máxima de 3:1")
    return problems


def check_size(model: Model, size: str) -> tuple[int, int]:
    """Valida ``size`` para o modelo e devolve (largura, altura)."""
    if size == "auto":
        raise ValueError("size='auto' não tem estimativa de custo; informe LARGURAxALTURA")
    width, height = parse_size(size)
    if not model.custom_sizes:
        if size not in FIXED_SIZES:
            raise ValueError(f"{model.name} só aceita {', '.join(FIXED_SIZES)} (recebido {size})")
    elif problems := size_problems(width, height):
        raise ValueError(f"tamanho {size} inválido para {model.name}: " + "; ".join(problems))
    return width, height


# ----------------------------------------------------------------------------- tokens e custo


def _js_round(x: float) -> int:
    """``Math.round`` do JavaScript (meio para cima), que a calculadora oficial usa."""
    return math.floor(x + 0.5)


def output_tokens(model: Model, quality: str, width: int, height: int) -> int | None:
    """Tokens de saída de uma imagem, pela fórmula da calculadora oficial.

    ``None`` para os modelos de tamanho fixo (1.5 e mini), que usam a tabela de US$ por imagem.
    """
    if not model.custom_sizes:
        return None
    base_table = _BASE_2 if model.name == "gpt-image-2" else _BASE_25
    if quality not in base_table:
        raise ValueError(f"{model.name} não aceita quality={quality!r}; opções: {', '.join(base_table)}")
    base = base_table[quality]
    longer, shorter = max(width, height), min(width, height)
    scaled = base / (longer / shorter)
    floor = math.floor(scaled)
    other = floor + floor % 2 if scaled - floor == 0.5 else _js_round(scaled)   # meio: arredonda para par
    cells = (base if width >= height else other) * (other if width >= height else base)
    tokens = math.ceil(cells * (2e6 + width * height) / 4e6)
    if model.name.startswith("gpt-image-2.6"):
        tokens = _js_round(tokens * _MULT_26[quality])
    return tokens


def prompt_tokens_estimate(prompt: str) -> int:
    """Estimativa grosseira de tokens de texto do prompt (~4 caracteres por token em inglês)."""
    return max(1, math.ceil(len(prompt) / 4))


@dataclass(frozen=True)
class Estimate:
    model: str
    quality: str
    size: str
    output_tokens: int | None
    output_usd: float
    input_usd: float
    batch_usd: float | None     # custo total no Batch API (metade); None se o modelo não tem Batch

    @property
    def total_usd(self) -> float:
        return self.output_usd + self.input_usd


def estimate(model_name: str, quality: str, size: str, prompt_tokens: int = 0) -> Estimate:
    model = get_model(model_name)
    if quality == "auto":
        raise ValueError("quality='auto' não tem estimativa: o modelo escolhe a qualidade")
    if quality not in model.qualities:
        raise ValueError(f"{model.name} não aceita quality={quality!r}; opções: {', '.join(model.qualities)}")
    width, height = check_size(model, size)
    tokens = output_tokens(model, quality, width, height)
    if tokens is None:
        index = FIXED_SIZES.index(size)
        output_usd = _LEGACY_USD[model.name][quality][index]
    else:
        output_usd = tokens * model.rates.image_out / 1e6
    input_usd = prompt_tokens * model.rates.text_in / 1e6
    batch = (output_usd + input_usd) * BATCH_DISCOUNT if model.batch else None
    return Estimate(model.name, quality, size, tokens, output_usd, input_usd, batch)


def cost_from_usage(model_name: str, usage: dict | None) -> float | None:
    """Custo real (US$) a partir do ``usage`` da resposta; ``None`` se a resposta não trouxe ``usage``."""
    if not usage:
        return None
    rates = get_model(model_name).rates
    in_details = usage.get("input_tokens_details") or {}
    out_details = usage.get("output_tokens_details") or {}
    image_in = in_details.get("image_tokens", 0)
    text_in = in_details.get("text_tokens", max(usage.get("input_tokens", 0) - image_in, 0))
    if "image_tokens" in out_details:
        image_out, text_out = out_details["image_tokens"], out_details.get("text_tokens", 0)
    else:
        image_out, text_out = usage.get("output_tokens", 0), 0
    return (text_in * rates.text_in + image_in * rates.image_in
            + image_out * rates.image_out + text_out * rates.text_out) / 1e6


def batch_cost_from_usage(model_name: str, usage: dict | None) -> float | None:
    """Custo real (US$) de uma imagem gerada pela Batch API: o do modo normal com o desconto do batch."""
    cost = cost_from_usage(model_name, usage)
    return None if cost is None else cost * BATCH_DISCOUNT

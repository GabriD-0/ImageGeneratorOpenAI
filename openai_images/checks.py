from __future__ import annotations

import io
import struct

DEFAULT_KEY_COLOR = (0x00, 0xB1, 0x40)   # #00B140
KEY_TOLERANCE = 60                        # distância euclidiana em RGB aceita como "o mesmo verde"
KEY_MIN_SHARE = 0.90                      # fração da borda que precisa ser chroma key para passar

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def parse_color(text: str) -> tuple[int, int, int]:
    value = text.strip().lstrip("#")

    if len(value) != 6:
        raise ValueError(f"cor inválida: {text!r} (use #RRGGBB)")

    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def png_size(data: bytes) -> tuple[int, int] | None:
    if data[:8] == _PNG_MAGIC and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])

    return None


def analyze(data: bytes, *, min_height: int = 1600, key_color: tuple[int, int, int] | None = DEFAULT_KEY_COLOR) -> dict:
    """Resumo objetivo da imagem. ``key_color=None`` desliga a checagem do fundo."""
    result: dict = {"bytes": len(data), "width": None, "height": None, "meets_height": None,
                    "key_share": None, "key_ok": None, "has_alpha": None}
    size = png_size(data)
    if size:
        result["width"], result["height"] = size
    try:
        from PIL import Image
    except ImportError:
        Image = None   # sem Pillow: só as dimensões do PNG
    if Image is not None:
        try:
            with Image.open(io.BytesIO(data)) as img:
                result["width"], result["height"] = img.size
                result["has_alpha"] = "A" in img.getbands()
                if key_color is not None:
                    result["key_share"] = _border_key_share(img, key_color)
        except OSError:
            pass
    if result["height"] is not None:
        result["meets_height"] = result["height"] >= min_height
    if result["key_share"] is not None:
        result["key_ok"] = result["key_share"] >= KEY_MIN_SHARE
    return result


def _border_key_share(img, key: tuple[int, int, int]) -> float:
    """Fração dos pixels da moldura externa (2% de cada lado) que estão perto do verde chroma key."""
    rgb = img.convert("RGB")
    width, height = rgb.size
    band = max(2, round(min(width, height) * 0.02))
    strips = [(0, 0, width, band), (0, height - band, width, height),
              (0, band, band, height - band), (width - band, band, width, height - band)]
    limit = KEY_TOLERANCE ** 2
    near = total = 0
    for box in strips:
        for r, g, b in rgb.crop(box).getdata():
            total += 1
            if (r - key[0]) ** 2 + (g - key[1]) ** 2 + (b - key[2]) ** 2 <= limit:
                near += 1
    return near / total if total else 0.0

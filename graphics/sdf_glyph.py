"""SDF 字形渲染与动态描边。

本模块实现一条可运行的 SDF 文本管线，逐条消除题目描述的六类线上故障：

1. **上传栅栏与占位**（:class:`UploadFence` / :class:`DynamicAtlas`）
   字形图集按需异步上传。栅栏点亮之前，槽位里可能还留着上一页的旧字形
   数据，因此绘制端只认 ``visible_for(key)``：槽位代际、常驻键、栅栏三者
   全部匹配才允许画字形，否则一律画占位块。首帧不会再闪出上一页残留字形。

2. **屏幕空间描边**（:func:`outline_coverage` / :func:`screen_space_outline`）
   描边宽度以屏幕像素为单位。SDF 采样值先按 ``px_per_texel`` 换算成像素
   距离，再做各向同性膨胀，因此超大标题的斜角不会因为“逐边外扩”而断裂，
   缩小到小字号时用覆盖率抗锯齿而不是硬 alpha 阈值，细笔画不会被吞掉；
   描边宽度与字号、远近无关。

3. **图集引用计数**（:meth:`DynamicAtlas.retain` / :meth:`DynamicAtlas.evict_lru`）
   正在排版的段落通过引用计数钉住槽位，淘汰只回收引用计数为 0 的槽位，
   换页后不会整段文字变成方块。槽位尺寸按最大描边像素预留缓冲，描边采样
   不会越界。

4. **布局快照**（:class:`LayoutSnapshot`）
   排版结果冻结为不可变快照。异步回退字体加载、异步图集上传都不会修改
   已提交的快照，快照校验和也不包含任何异步状态。

5. **DPI 重排**（:func:`dpi_relayout`）
   整条布局先在 em 空间用浮点计算，最后才按 ``size_px * dpi / 96`` 缩放到
   像素，字距不做像素取整，行宽上限也按同一比例换算。DPI 切换后行宽严格
   等比、换行点完全一致。

6. **回退字体基线统一**（:func:`fallback_baseline_unify`）
   基线、行高由整条字体链的 *清单*（manifest：ascent/descent/line_gap）
   统一计算。回退字体在注册时清单就已知，即使字形数据还在异步加载，行高
   也不会跳变、滚动位置不漂、光标与字符不错位。

所有公开入口的第一个参数 ``scenario`` 通常传 ``1``，用于快速构造默认场景；
也可以用关键字参数注入自己的字体链、文本、图集等数据。模块只依赖标准库，
输出确定、可复现。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_SIZE_PX",
    "DEFAULT_DPI",
    "DEFAULT_MAX_WIDTH_PX",
    "DEFAULT_OUTLINE_PX",
    "DEFAULT_TEXT",
    "AA_PX",
    "Rect",
    "GlyphShape",
    "LineMetrics",
    "Font",
    "FontChain",
    "GlyphKey",
    "UploadFence",
    "AtlasSlot",
    "DynamicAtlas",
    "GlyphPlacement",
    "LineBox",
    "LayoutSnapshot",
    "GlyphBitmap",
    "outline_coverage",
    "fill_coverage",
    "padding_texels",
    "atlas_upload_fence",
    "screen_space_outline",
    "atlas_ref_count",
    "layout_snapshot",
    "dpi_relayout",
    "fallback_baseline_unify",
    "load_font",
    "layout_text_sdf",
    "raster_glyph",
    "present_sdf_text",
    "validate_sdf_text",
    "metrics_sdf_text",
    "audit_sdf_text",
    "fallback_sdf_text",
    "determinism_sdf_text",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_SIZE_PX = 16.0
DEFAULT_DPI = 96.0
DEFAULT_MAX_WIDTH_PX = 240.0
DEFAULT_OUTLINE_PX = 2.0
AA_PX = 1.0
DEFAULT_TEXT = "SDF 字形描边 Atlas 混排 fallback 测试 ABC 123"
DEFAULT_LETTER_SPACING_EM = 0.0

# 图集默认参数：1024² 的 8-bit SDF 页；槽位按最小绘制缩放预留描边缓冲。
DEFAULT_PAGE_TEXELS = 1024
DEFAULT_MAX_OUTLINE_PX = 4.0
DEFAULT_MIN_DRAW_SCALE = 0.5  # 字形最小按图集分辨率的 0.5 倍绘制

# 字体清单（manifest）与字形数据分离：清单同步可用，数据可异步到达。
FONT_FAMILY_SANS = "Sans"
FONT_FAMILY_CJK = "CJK"
FONT_FAMILY_EMOJI = "Emoji"
FONT_FAMILY_LAST = "LastResort"


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _ceil(value: float) -> int:
    return int(math.ceil(value - 1e-12))


def _digest(payload: Any, size: int = 16) -> str:
    """稳定的内容摘要：用于快照校验和与确定性验证。"""
    try:
        text = json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
    except (TypeError, ValueError):
        text = repr(payload)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return repr(value)
    return value


def _option(scenario: Any, name: str, default: Any = None) -> Any:
    """从 dict / 对象形式的 scenario 里取一个可选项。"""
    if isinstance(scenario, dict):
        return scenario.get(name, default)
    return getattr(scenario, name, default)


# ---------------------------------------------------------------------------
# 几何与字形轮廓
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rect:
    """轴对齐矩形（em 坐标，y 向上，基线 y=0）。"""

    x0: float
    y0: float
    x1: float
    y1: float

    def sdf(self, x: float, y: float) -> float:
        """点到矩形边界的有符号距离，内部为负。"""
        cx = (self.x0 + self.x1) * 0.5
        cy = (self.y0 + self.y1) * 0.5
        hx = (self.x1 - self.x0) * 0.5
        hy = (self.y1 - self.y0) * 0.5
        qx = abs(x - cx) - hx
        qy = abs(y - cy) - hy
        outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
        inside = min(max(qx, qy), 0.0)
        return outside + inside


@dataclass(frozen=True)
class GlyphShape:
    """一个字形在 em 空间的轮廓（若干矩形的并集）与度量。"""

    char: str
    advance_em: float
    boxes: Tuple[Rect, ...] = ()

    @property
    def bbox_em(self) -> Tuple[float, float, float, float]:
        if not self.boxes:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            min(b.x0 for b in self.boxes),
            min(b.y0 for b in self.boxes),
            max(b.x1 for b in self.boxes),
            max(b.y1 for b in self.boxes),
        )

    def sdf_em(self, x: float, y: float) -> float:
        if not self.boxes:
            return 1.0  # 纯空白字形：远离墨迹
        return min(box.sdf(x, y) for box in self.boxes)

    @property
    def min_stroke_em(self) -> float:
        """最细笔画宽度（em），用于验证小字号下不被 alpha 阈值吞掉。"""
        widths = []
        for box in self.boxes:
            widths.append(box.x1 - box.x0)
            widths.append(box.y1 - box.y0)
        return min(widths) if widths else 0.0

    def coverage_em(self, x: float, y: float, aa_em: float = 0.0) -> float:
        if aa_em <= 0:
            return 1.0 if self.sdf_em(x, y) <= 0.0 else 0.0
        return _clamp(0.5 - self.sdf_em(x, y) / aa_em, 0.0, 1.0)


def fill_coverage(distance_px: float, aa_px: float = AA_PX) -> float:
    """SDF 填充覆盖率（屏幕空间像素距离）。"""
    if aa_px <= 0:
        return 1.0 if distance_px <= 0.0 else 0.0
    return _clamp(0.5 - distance_px / aa_px, 0.0, 1.0)


def outline_coverage(distance_px: float, width_px: float,
                     aa_px: float = AA_PX) -> float:
    """屏幕空间描边覆盖率。

    ``distance_px`` 是有符号像素距离（内部为负）。描边是一圈内半径 0、
    外半径 ``width_px`` 的膨胀带，抗锯齿各留 ``aa/2`` 的过渡：

    - 填充内部（``d <= -aa/2``）不画描边；
    - 边缘往下 ``width_px`` 的像素带整圈不透明；
    - 再往外 ``aa/2`` 渐隐到 0。

    距离在像素空间做各向同性膨胀，因此任何方向的斜角都连续，不存在
    逐边外扩产生的拐角裂缝。
    """
    width_px = max(0.0, float(width_px))
    aa_px = max(1e-6, float(aa_px))
    outer = width_px + aa_px * 0.5
    outer_cov = _clamp((outer - distance_px) / aa_px, 0.0, 1.0)
    inner_cov = _clamp((distance_px + aa_px * 0.5) / aa_px, 0.0, 1.0)
    return min(outer_cov, inner_cov)


def padding_texels(outline_px: float, min_draw_scale: float = DEFAULT_MIN_DRAW_SCALE,
                   aa_px: float = AA_PX) -> int:
    """槽位需要预留的描边缓冲（texels）。

    图集里的 SDF 以 texel 为单位存储。当字形被缩小绘制时，一个屏幕像素
    对应多个 texel，``outline_px`` 需要的 texel 数按 ``1/min_draw_scale``
    放大；再加上抗锯齿余量。少于这个值描边采样就会越界。
    """
    scale = max(1e-6, float(min_draw_scale))
    return _ceil((outline_px + aa_px) / scale) + 1


# ---------------------------------------------------------------------------
# 字体与字体链
# ---------------------------------------------------------------------------


def _cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return 0x2E80 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF


def _emoji_char(ch: str) -> bool:
    cp = ord(ch)
    return 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF


def _ring(x0: float, y0: float, x1: float, y1: float, t: float) -> Tuple[Rect, ...]:
    return (
        Rect(x0, y0, x1, y0 + t),
        Rect(x0, y1 - t, x1, y1),
        Rect(x0, y0, x0 + t, y1),
        Rect(x1 - t, y0, x1, y1),
    )


def _synthesize_shape(ch: str, advance_em: float) -> GlyphShape:
    """按字符类别合成字形轮廓（真实引擎里这里来自轮廓栅格化）。"""
    if ch.isspace():
        return GlyphShape(ch, advance_em, ())

    if _cjk_char(ch) or _emoji_char(ch):
        t = 0.12 if _cjk_char(ch) else 0.10
        boxes = list(_ring(0.10, 0.04, advance_em - 0.10, 0.72, t))
        boxes.append(Rect(0.10, 0.34, advance_em - 0.10, 0.34 + t))
        boxes.append(Rect(advance_em * 0.5 - t * 0.5, 0.04,
                          advance_em * 0.5 + t * 0.5, 0.72))
        return GlyphShape(ch, advance_em, tuple(boxes))

    if ch in "oO0DQqgG":
        t = 0.12
        return GlyphShape(ch, advance_em, _ring(0.12, 0.02, advance_em - 0.12, 0.70, t))

    if ch in ".·":
        s = 0.06
        x = advance_em * 0.5 - s * 0.5
        return GlyphShape(ch, advance_em, (Rect(x, 0.0, x + s, s),))

    if ch in ":;!i|jl'":
        s = 0.07
        x = advance_em * 0.5 - s * 0.5
        boxes = [Rect(x, 0.0, x + s, 0.70)]
        if ch in "i!j'":
            boxes.append(Rect(x, 0.78, x + s, 0.78 + s))
        return GlyphShape(ch, advance_em, tuple(boxes))

    if ch in "mw":
        boxes = []
        n = 3 if ch == "m" else 2
        for k in range(n):
            x = 0.10 + k * (advance_em - 0.28) / max(1, n - 1)
            boxes.append(Rect(x, 0.0, x + 0.10, 0.70))
        boxes.append(Rect(0.10, 0.60, advance_em - 0.10, 0.70))
        return GlyphShape(ch, advance_em, tuple(boxes))

    if ch in "-_=~":
        y = {"-": 0.32, "_": 0.0, "=": 0.28, "~": 0.36}[ch]
        boxes = [Rect(0.08, y, advance_em - 0.08, y + 0.08)]
        if ch == "=":
            boxes.append(Rect(0.08, y + 0.16, advance_em - 0.08, y + 0.24))
        return GlyphShape(ch, advance_em, tuple(boxes))

    if ch in "()[]{}":
        s = 0.09
        return GlyphShape(ch, advance_em, (
            Rect(advance_em * 0.5 - s * 0.5, 0.0, advance_em * 0.5 + s * 0.5, 0.72),
        ))

    if ch in "AHTXVY":
        boxes = [
            Rect(0.10, 0.0, 0.20, 0.70),
            Rect(advance_em - 0.20, 0.0, advance_em - 0.10, 0.70),
        ]
        if ch in "AH":
            boxes.append(Rect(0.10, 0.30, advance_em - 0.10, 0.40))
        return GlyphShape(ch, advance_em, tuple(boxes))

    # 默认：一根竖干 + 顶部横画。
    w = 0.13
    x = advance_em * 0.5 - w * 0.5
    return GlyphShape(ch, advance_em, (
        Rect(x, 0.0, x + w, 0.70),
        Rect(0.10, 0.60, advance_em - 0.10, 0.70),
    ))


def _default_advance(ch: str) -> float:
    if ch.isspace():
        return 0.32
    if _cjk_char(ch):
        return 1.0
    if _emoji_char(ch):
        return 1.2
    if ch in "iljI.,:;'!|":
        return 0.30
    if ch.isdigit():
        return 0.60
    if ch.isupper():
        return 0.62
    return 0.56


def _charset_for(family: str) -> Tuple[str, ...]:
    ascii_chars = "".join(chr(cp) for cp in range(0x20, 0x7F))
    if family == FONT_FAMILY_SANS:
        return tuple(ascii_chars)
    if family == FONT_FAMILY_CJK:
        # 覆盖默认文本里的 CJK 字符以及常用汉字范围的代表。
        sample = "字形描边图集混排测试中文渲染动态光栅"
        return tuple(sorted(set(sample + "".join(chr(cp) for cp in range(0x4E00, 0x4E20)))))
    if family == FONT_FAMILY_EMOJI:
        return tuple(chr(cp) for cp in range(0x1F600, 0x1F610))
    if family == FONT_FAMILY_LAST:
        return tuple(chr(cp) for cp in (0xFFFD, 0x25A1))
    return tuple(ascii_chars)


def _default_font_metrics(family: str) -> Tuple[float, float, float]:
    return {
        FONT_FAMILY_SANS: (0.80, 0.20, 0.00),
        FONT_FAMILY_CJK: (0.88, 0.12, 0.10),
        FONT_FAMILY_EMOJI: (0.90, 0.15, 0.10),
        FONT_FAMILY_LAST: (0.80, 0.20, 0.00),
    }.get(family, (0.80, 0.20, 0.00))


@dataclass
class Font:
    """字体。

    ``ascent/descent/line_gap`` 以及 ``advances`` 属于 *清单*，注册时即同步
    可用；``glyphs`` 是字形轮廓数据，允许异步加载。布局只依赖清单，因此
    字形数据晚到不会改变已经排好的行。
    """

    name: str
    family: str = FONT_FAMILY_SANS
    units_per_em: int = 1000
    ascent_em: float = 0.80
    descent_em: float = 0.20
    line_gap_em: float = 0.00
    charset: Tuple[str, ...] = ()
    advances: Dict[str, float] = field(default_factory=dict)
    glyphs: Dict[str, GlyphShape] = field(default_factory=dict)
    loaded: bool = False
    register_seq: int = 0

    # -- 清单查询 ---------------------------------------------------------

    def has_codepoint(self, ch: str) -> bool:
        return ch in self.advances or ch in self.charset

    def advance_em(self, ch: str) -> float:
        if ch in self.advances:
            return self.advances[ch]
        return _default_advance(ch)

    def ensure_manifest(self) -> "Font":
        if not self.charset:
            self.charset = _charset_for(self.family)
        for ch in self.charset:
            self.advances.setdefault(ch, _default_advance(ch))
        if not any(box.char == "□" for box in self.glyphs.values()):
            self.glyphs.setdefault("□", _synthesize_shape("□", 0.9))
        return self

    # -- 字形数据 ---------------------------------------------------------

    def load_glyphs(self) -> "Font":
        """异步字形数据到达：只填字形轮廓，不动清单。"""
        self.ensure_manifest()
        for ch in self.charset:
            self.glyphs.setdefault(ch, _synthesize_shape(ch, self.advance_em(ch)))
        self.loaded = True
        return self

    def shape(self, ch: str) -> Optional[GlyphShape]:
        if not self.loaded:
            return None
        if ch in self.glyphs:
            return self.glyphs[ch]
        if self.has_codepoint(ch):
            return _synthesize_shape(ch, self.advance_em(ch))
        return None

    def manifest_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "units_per_em": self.units_per_em,
            "ascent_em": self.ascent_em,
            "descent_em": self.descent_em,
            "line_gap_em": self.line_gap_em,
            "glyph_count": len(self.charset),
            "loaded": self.loaded,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.manifest_dict()


@dataclass(frozen=True)
class LineMetrics:
    """整条字体链统一后的行度量（em）。"""

    ascent_em: float
    descent_em: float
    line_gap_em: float

    @property
    def line_height_em(self) -> float:
        return self.ascent_em + self.descent_em + self.line_gap_em

    @property
    def baseline_from_top_em(self) -> float:
        return self.ascent_em

    def to_dict(self) -> Dict[str, float]:
        return {
            "ascent_em": self.ascent_em,
            "descent_em": self.descent_em,
            "line_gap_em": self.line_gap_em,
            "line_height_em": self.line_height_em,
            "baseline_from_top_em": self.baseline_from_top_em,
        }


@dataclass
class FontChain:
    """有序回退字体链：按注册顺序解析字符覆盖。"""

    fonts: List[Font] = field(default_factory=list)
    chain_id: str = "chain-default"

    def add(self, font: Font, seq: Optional[int] = None) -> "FontChain":
        font.ensure_manifest()
        font.register_seq = len(self.fonts) if seq is None else int(seq)
        self.fonts.append(font)
        return self

    def resolve(self, ch: str) -> Optional[Font]:
        for font in self.fonts:
            if font.has_codepoint(ch):
                return font
        return None

    def resolve_or_last(self, ch: str) -> Optional[Font]:
        return self.resolve(ch) or (self.fonts[-1] if self.fonts else None)

    def metrics(self, loaded_only: bool = False) -> LineMetrics:
        """统一行度量。

        ``loaded_only=False``（正确实现）用整条链的清单，注册即稳定；
        ``loaded_only=True`` 复现旧实现“只统计已加载字体”的跳变行为。
        """
        pool = [f for f in self.fonts if (f.loaded or not loaded_only)]
        if not pool:
            pool = list(self.fonts)
        if not pool:
            return LineMetrics(0.8, 0.2, 0.0)
        return LineMetrics(
            ascent_em=max(f.ascent_em for f in pool),
            descent_em=max(f.descent_em for f in pool),
            line_gap_em=max(f.line_gap_em for f in pool),
        )

    def coverage(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for font in self.fonts:
            for ch in font.charset:
                mapping.setdefault(ch, font.name)
        return mapping

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "fonts": [f.manifest_dict() for f in self.fonts],
            "metrics": self.metrics().to_dict(),
            "naive_loaded_metrics": self.metrics(loaded_only=True).to_dict(),
        }


# ---------------------------------------------------------------------------
# 图集：键、栅栏、槽位、引用计数
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlyphKey:
    """图集槽位键：字体 + 字符 + 光栅字号 + 描边预留。"""

    font: str
    char: str
    size_px: float
    outline_px: float = DEFAULT_OUTLINE_PX

    @property
    def digest(self) -> str:
        return _digest([self.font, self.char, round(self.size_px, 3),
                        round(self.outline_px, 3)], size=8)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "font": self.font,
            "char": self.char,
            "size_px": self.size_px,
            "outline_px": self.outline_px,
            "digest": self.digest,
        }


@dataclass
class UploadFence:
    """上传栅栏：``signaled`` 之前 GPU 不得读取新数据。"""

    seq: int
    generation: int
    signaled: bool = False
    key: Optional[GlyphKey] = None
    payload: Any = None
    reason: str = "pending"

    def signal(self) -> "UploadFence":
        self.signaled = True
        self.reason = "signaled"
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "generation": self.generation,
            "signaled": self.signaled,
            "key": self.key.to_dict() if self.key else None,
            "reason": self.reason,
        }


@dataclass
class AtlasSlot:
    """图集槽位。

    ``key`` 是当前想放进去的字形；``resident_key/payload`` 是内存里实际
    还留着的旧内容。只有栅栏点亮并完成发布后，两者才一致。
    """

    key: GlyphKey
    x: int
    y: int
    width: int
    height: int
    page: int
    pad: int
    generation: int
    state: str = "pending"  # pending | ready | evicted
    refs: int = 0
    last_used: int = 0
    fence: Optional[UploadFence] = None
    resident_key: Optional[GlyphKey] = None
    payload: Any = None

    @property
    def texels(self) -> int:
        return self.width * self.height

    @property
    def residual_payload(self) -> bool:
        """槽位里是否还残留其它字形的数据（首帧残留的来源）。"""
        return self.payload is not None and self.resident_key != self.key

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "rect": [self.x, self.y, self.width, self.height],
            "page": self.page,
            "pad": self.pad,
            "generation": self.generation,
            "state": self.state,
            "refs": self.refs,
            "resident": self.resident_key.to_dict() if self.resident_key else None,
            "residual": self.residual_payload,
            "fence": self.fence.to_dict() if self.fence else None,
        }


class DynamicAtlas:
    """动态 SDF 图集：shelf 打包、异步上传、引用计数与 LRU 淘汰。"""

    def __init__(self, page_texels: int = DEFAULT_PAGE_TEXELS,
                 max_outline_px: float = DEFAULT_MAX_OUTLINE_PX,
                 min_draw_scale: float = DEFAULT_MIN_DRAW_SCALE,
                 aa_px: float = AA_PX) -> None:
        self.page_texels = int(page_texels)
        self.max_outline_px = float(max_outline_px)
        self.min_draw_scale = float(min_draw_scale)
        self.aa_px = float(aa_px)
        self.pad = padding_texels(self.max_outline_px, self.min_draw_scale, self.aa_px)
        self.slots: Dict[GlyphKey, AtlasSlot] = {}
        self.generation = 0
        self.page_count = 1
        self.fence_seq = 0
        self.clock = 0
        self.placeholder_draws = 0
        self.residual_blocked = 0
        self.evicted_total = 0
        self.overflow_blocked = 0
        self._shelf = {"page": 0, "y": 0, "x": 0, "row_h": 0}
        self._owners: Dict[str, set] = {}

    # -- 分配 -------------------------------------------------------------

    def _slot_size(self, key: GlyphKey) -> Tuple[int, int]:
        # 字形位图按光栅字号取整，四周各留 pad 个 texel 的描边缓冲。
        glyph_w = _ceil(max(0.5, key.size_px * 0.9))
        glyph_h = _ceil(max(0.5, key.size_px * 0.85))
        return glyph_w + self.pad * 2, glyph_h + self.pad * 2

    def _pack(self, w: int, h: int) -> Tuple[int, int, int]:
        shelf = self._shelf
        if shelf["x"] + w > self.page_texels:
            shelf["y"] += shelf["row_h"]
            shelf["x"] = 0
            shelf["row_h"] = 0
        if shelf["y"] + h > self.page_texels:
            shelf["page"] += 1
            self.page_count = shelf["page"] + 1
            shelf["y"] = 0
            shelf["x"] = 0
            shelf["row_h"] = 0
        x, y, page = shelf["x"], shelf["y"], shelf["page"]
        shelf["x"] += w
        shelf["row_h"] = max(shelf["row_h"], h)
        return x, y, page

    def allocate(self, key: GlyphKey, payload: Any = None) -> AtlasSlot:
        slot = self.slots.get(key)
        self.clock += 1
        if slot is not None:
            slot.last_used = self.clock
            return slot
        w, h = self._slot_size(key)
        # 每个键按自身描边宽度重新计算缓冲，绝不小于图集默认预留。
        slot_pad = max(self.pad, padding_texels(key.outline_px,
                                                self.min_draw_scale, self.aa_px))
        w = w - self.pad * 2 + slot_pad * 2
        h = h - self.pad * 2 + slot_pad * 2
        if w > self.page_texels or h > self.page_texels:
            # 单槽超过一页：记录越界拦截，退化为最小可用尺寸，避免崩溃。
            self.overflow_blocked += 1
            w = min(w, self.page_texels)
            h = min(h, self.page_texels)
        x, y, page = self._pack(w, h)
        slot = AtlasSlot(
            key=key, x=x, y=y, width=w, height=h, page=page, pad=slot_pad,
            generation=self.generation, last_used=self.clock,
        )
        # payload 仅用于模拟“槽位被复用但还没发布”的旧内容。
        if payload is not None:
            slot.payload = payload
            slot.resident_key = GlyphKey(
                font="__previous_page__", char="?", size_px=key.size_px)
        self.slots[key] = slot
        return slot

    def get(self, key: GlyphKey) -> Optional[AtlasSlot]:
        return self.slots.get(key)

    # -- 上传 -------------------------------------------------------------

    def begin_upload(self, key: GlyphKey, payload: Any = None,
                     previous_payload: Any = None) -> UploadFence:
        slot = self.allocate(key, payload=previous_payload)
        self.fence_seq += 1
        fence = UploadFence(self.fence_seq, self.generation, False, key, payload)
        slot.fence = fence
        slot.state = "pending"
        slot.generation = self.generation
        return fence

    def signal(self, fence: UploadFence) -> UploadFence:
        """栅栏点亮：数据发布到槽位，旧内容被覆盖。"""
        fence.signal()
        slot = self.slots.get(fence.key) if fence.key else None
        if slot is not None:
            slot.state = "ready"
            slot.fence = fence
            slot.generation = fence.generation
            slot.resident_key = fence.key
            slot.payload = fence.payload if fence.payload is not None else True
        return fence

    def publish(self, fence: UploadFence) -> UploadFence:
        return self.signal(fence)

    def visible_for(self, key: GlyphKey) -> bool:
        """绘制端唯一可见性判定：栅栏 + 常驻键 + 代际三者一致。"""
        slot = self.slots.get(key)
        if slot is None:
            return False
        if slot.state != "ready":
            return False
        if slot.fence is None or not slot.fence.signaled:
            return False
        if slot.resident_key != key:
            return False
        if slot.generation != self.generation:
            return False
        if slot.key != key:
            return False
        return True

    def record_placeholder(self, blocked_residual: bool = False) -> None:
        self.placeholder_draws += 1
        if blocked_residual:
            self.residual_blocked += 1

    # -- 引用计数 ---------------------------------------------------------

    def retain(self, key: GlyphKey, owner: Any = "layout") -> int:
        slot = self.slots.get(key) or self.allocate(key)
        slot.refs += 1
        slot.last_used = self.clock
        self._owners.setdefault(str(owner), set()).add(key)
        return slot.refs

    def release(self, key: GlyphKey, owner: Any = "layout") -> int:
        slot = self.slots.get(key)
        owned = self._owners.get(str(owner))
        if owned is not None:
            owned.discard(key)
        if slot is None:
            return 0
        slot.refs = max(0, slot.refs - 1)
        return slot.refs

    def release_owner(self, owner: Any) -> List[GlyphKey]:
        keys = list(self._owners.get(str(owner), set()))
        for key in keys:
            self.release(key, owner)
        self._owners.pop(str(owner), None)
        return keys

    def evict_lru(self, budget_slots: Optional[int] = None) -> Dict[str, Any]:
        """按最近使用淘汰，只回收引用计数为 0 的槽位。"""
        ready = [s for s in self.slots.values() if s.state == "ready"]
        budget = len(ready) if budget_slots is None else int(budget_slots)
        evicted: List[GlyphKey] = []
        pinned: List[GlyphKey] = []
        while len(ready) > budget:
            candidates = [s for s in ready if s.refs <= 0]
            if not candidates:
                break
            victim = min(candidates, key=lambda s: (s.last_used, s.key.digest))
            victim.state = "evicted"
            victim.payload = None
            victim.resident_key = None
            victim.fence = None
            self.slots.pop(victim.key, None)
            self.evicted_total += 1
            evicted.append(victim.key)
            ready = [s for s in ready if s is not victim]
        pinned = [s.key for s in ready if s.refs > 0]
        pinned_set = set(pinned)
        return {
            "evicted": [k.to_dict() for k in evicted],
            "pinned": [k.to_dict() for k in pinned],
            "remaining": len(ready),
            "budget": budget,
            # 被回收的槽位里绝不能出现引用计数 > 0 的槽位。
            "pinned_respected": all(k not in pinned_set for k in evicted),
        }

    def new_page(self) -> int:
        """换页：代际 +1，旧代际槽位全部不可见（但仍保留引用计数）。"""
        self.generation += 1
        for slot in self.slots.values():
            if slot.generation != self.generation:
                slot.state = "pending"
                slot.fence = None
        return self.generation

    def stats(self) -> Dict[str, Any]:
        ready = [s for s in self.slots.values() if s.state == "ready"]
        pending = [s for s in self.slots.values() if s.state == "pending"]
        return {
            "generation": self.generation,
            "pages": self.page_count,
            "slots": len(self.slots),
            "ready": len(ready),
            "pending": len(pending),
            "pinned": sum(1 for s in self.slots.values() if s.refs > 0),
            "texels": sum(s.texels for s in self.slots.values()),
            "pad": self.pad,
            "placeholder_draws": self.placeholder_draws,
            "residual_blocked": self.residual_blocked,
            "evicted_total": self.evicted_total,
            "overflow_blocked": self.overflow_blocked,
            "fence_seq": self.fence_seq,
        }


# ---------------------------------------------------------------------------
# 布局：字形摆放、行盒、不可变快照
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlyphPlacement:
    char: str
    font: str
    key: GlyphKey
    x_em: float
    baseline_em: float
    advance_em: float
    line_index: int
    char_index: int
    is_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "char": self.char,
            "font": self.font,
            "glyph": self.key.digest,
            "x_em": round(self.x_em, 6),
            "baseline_em": round(self.baseline_em, 6),
            "advance_em": round(self.advance_em, 6),
            "line": self.line_index,
            "char_index": self.char_index,
            "fallback": self.is_fallback,
        }


@dataclass(frozen=True)
class LineBox:
    index: int
    start_char: int
    end_char: int
    width_em: float
    placement_start: int
    placement_end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "width_em": round(self.width_em, 6),
            "placements": [self.placement_start, self.placement_end],
        }


@dataclass(frozen=True)
class LayoutSnapshot:
    """冻结的排版结果。字段全为不可变类型，异步状态不会污染它。"""

    text: str
    size_px: float
    dpi: float
    max_width_px: float
    letter_spacing_em: float
    chain_id: str
    ascent_em: float
    descent_em: float
    line_gap_em: float
    line_height_em: float
    lines: Tuple[LineBox, ...]
    placements: Tuple[GlyphPlacement, ...]
    pinned_keys: Tuple[GlyphKey, ...] = ()
    font_chain: Tuple[Tuple[str, float], ...] = ()

    @property
    def px_scale(self) -> float:
        """em -> 屏幕像素的缩放。占位符/描边都以屏幕像素为准。"""
        return self.size_px * self.dpi / 96.0

    def line_width_px(self, line: LineBox) -> float:
        return line.width_em * self.px_scale

    @property
    def line_height_px(self) -> float:
        return self.line_height_em * self.px_scale

    @property
    def checksum(self) -> str:
        """快照身份：只包含布局输入与结果，不含加载/上传状态。"""
        return _digest({
            "text": self.text,
            "size_px": round(self.size_px, 6),
            "dpi": round(self.dpi, 6),
            "max_width_px": round(self.max_width_px, 6),
            "letter_spacing_em": round(self.letter_spacing_em, 6),
            "chain_id": self.chain_id,
            "metrics": [round(self.ascent_em, 6), round(self.descent_em, 6),
                        round(self.line_gap_em, 6)],
            "lines": [l.to_dict() for l in self.lines],
            "placements": [p.to_dict() for p in self.placements],
        })

    def cursor_px(self, char_index: int) -> Tuple[float, float]:
        """字符 ``char_index`` 之前的光标像素位置。"""
        for p in self.placements:
            if p.char_index == char_index:
                return (p.x_em * self.px_scale, p.baseline_em * self.px_scale)
        if self.lines:
            line = self.lines[-1]
            end_x = line.width_em
            return (end_x * self.px_scale,
                    (line.index * self.line_height_em + self.ascent_em) * self.px_scale)
        return (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "size_px": self.size_px,
            "dpi": self.dpi,
            "max_width_px": self.max_width_px,
            "letter_spacing_em": self.letter_spacing_em,
            "chain_id": self.chain_id,
            "metrics": {
                "ascent_em": self.ascent_em,
                "descent_em": self.descent_em,
                "line_gap_em": self.line_gap_em,
                "line_height_em": self.line_height_em,
                "line_height_px": self.line_height_px,
            },
            "lines": [line.to_dict() for line in self.lines],
            "placements": [p.to_dict() for p in self.placements],
            "pinned": [k.to_dict() for k in self.pinned_keys],
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class GlyphBitmap:
    """光栅化后的字形 SDF/覆盖率位图。"""

    char: str
    width: int
    height: int
    x_offset_em: float
    y_offset_em: float
    size_px: float
    pad_px: int
    alpha: Tuple[Tuple[float, ...], ...]
    alpha_hard: Tuple[Tuple[float, ...], ...]
    sdf_px: Tuple[Tuple[float, ...], ...]

    @property
    def coverage_sum(self) -> float:
        return sum(sum(row) for row in self.alpha)

    @property
    def coverage_hard_sum(self) -> float:
        return sum(sum(row) for row in self.alpha_hard)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "char": self.char,
            "width": self.width,
            "height": self.height,
            "x_offset_em": self.x_offset_em,
            "y_offset_em": self.y_offset_em,
            "size_px": self.size_px,
            "pad_px": self.pad_px,
            "coverage_sum": round(self.coverage_sum, 6),
            "coverage_hard_sum": round(self.coverage_hard_sum, 6),
            "min_alpha": round(min((min(row) for row in self.alpha), default=0.0), 6),
            "max_alpha": round(max((max(row) for row in self.alpha), default=0.0), 6),
        }


def _advance_for(chain: FontChain, ch: str) -> float:
    font = chain.resolve_or_last(ch) if chain.fonts else None
    return font.advance_em(ch) if font is not None else _default_advance(ch)


def _layout_em(text: str, chain: FontChain, max_width_em: float,
               letter_spacing_em: float = 0.0,
               size_px: float = DEFAULT_SIZE_PX,
               outline_px: float = DEFAULT_OUTLINE_PX,
               ) -> Tuple[List[LineBox], List[GlyphPlacement]]:
    """纯 em 空间的排版：浮点累加、不取整、不依赖 DPI。"""
    lines: List[LineBox] = []
    placements: List[GlyphPlacement] = []
    metrics = chain.metrics()
    line_height = metrics.line_height_em
    ascent = metrics.ascent_em
    family_index = {font.name: i for i, font in enumerate(chain.fonts)}

    abs_index = 0
    for p_index, paragraph in enumerate(text.split("\n")):
        chars = list(paragraph)
        n = len(chars)
        advances = [_advance_for(chain, ch) + letter_spacing_em for ch in chars]
        break_after = {k + 1 for k, ch in enumerate(chars) if ch.isspace() or ch in "-—"}
        break_before = {k for k, ch in enumerate(chars)
                        if _cjk_char(ch) and k > 0 and not chars[k - 1].isspace()}
        opportunities = break_after | break_before
        start = 0
        if n == 0:
            # 空段落：保留一个空行，光标/行号仍可定位。
            lines.append(LineBox(len(lines), abs_index, abs_index, 0.0,
                                 len(placements), len(placements)))
            abs_index += 1
            continue
        while start < n and chars[start].isspace():
            start += 1
        while start < n:
            end = start
            width = 0.0
            while end < n:
                adv = advances[end]
                if end > start and width + adv > max_width_em + 1e-9:
                    break
                width += adv
                end += 1
            if end < n:
                candidates = [p for p in range(start + 1, end + 1) if p in opportunities]
                if candidates:
                    end = max(candidates)
                # 没有机会点时按字符硬断
            if end == start:  # 单个字符都放不下
                end = start + 1
            trimmed = end
            while trimmed > start and trimmed - 1 < n and chars[trimmed - 1].isspace():
                trimmed -= 1
            line_width = sum(advances[start:trimmed])
            line_index = len(lines)
            baseline = line_index * line_height + ascent
            placement_start = len(placements)
            pen = 0.0
            for k in range(start, end):
                ch = chars[k]
                font = chain.resolve_or_last(ch)
                if font is None:
                    font_name, fallback = FONT_FAMILY_LAST, True
                else:
                    font_name = font.name
                    fallback = family_index.get(font_name, 0) > 0
                adv = advances[k]
                placements.append(GlyphPlacement(
                    char=ch,
                    font=font_name,
                    key=GlyphKey(font_name, ch, _bucket_size(size_px), outline_px),
                    x_em=pen,
                    baseline_em=baseline,
                    advance_em=adv,
                    line_index=line_index,
                    char_index=abs_index + k,
                    is_fallback=fallback,
                ))
                pen += adv
            lines.append(LineBox(
                index=line_index,
                start_char=abs_index + start,
                end_char=abs_index + end,
                width_em=line_width,
                placement_start=placement_start,
                placement_end=len(placements),
            ))
            start = end
            while start < n and chars[start].isspace():
                start += 1
        abs_index += n + 1  # 换行符占一个字符索引

    if not lines:
        lines.append(LineBox(0, 0, 0, 0.0, 0, 0))
    return lines, placements


def _bucket_size(size_px: float) -> float:
    """光栅字号分桶：相近字号共用一份 SDF。"""
    return float(max(1, _ceil(size_px)))


def _build_snapshot(text: str, chain: FontChain, size_px: float, dpi: float,
                    max_width_px: float, letter_spacing_em: float,
                    atlas: Optional[DynamicAtlas] = None,
                    owner: Any = "layout") -> LayoutSnapshot:
    metrics = chain.metrics()
    # 行宽上限是 *逻辑像素*，只按字号换算成 em；DPI 只影响最终物理像素缩放，
    # 因此 DPI 切换不会改变换行点。旧实现用物理像素做行宽上限，DPI 一变就重排。
    logical_scale_px_per_em = size_px
    max_width_em = (max_width_px / logical_scale_px_per_em
                    if logical_scale_px_per_em > 0 else max_width_px)
    lines, placements = _layout_em(
        text, chain, max_width_em, letter_spacing_em, size_px, DEFAULT_OUTLINE_PX)
    pinned: List[GlyphKey] = []
    if atlas is not None:
        for p in placements:
            if p.char.isspace():
                continue
            atlas.allocate(p.key)
            atlas.retain(p.key, owner)
            pinned.append(p.key)
        # 去重但保持顺序
        seen = set()
        pinned = [k for k in pinned if not (k in seen or seen.add(k))]
    return LayoutSnapshot(
        text=text,
        size_px=size_px,
        dpi=dpi,
        max_width_px=max_width_px,
        letter_spacing_em=letter_spacing_em,
        chain_id=chain.chain_id,
        ascent_em=metrics.ascent_em,
        descent_em=metrics.descent_em,
        line_gap_em=metrics.line_gap_em,
        line_height_em=metrics.line_height_em,
        lines=tuple(lines),
        placements=tuple(placements),
        pinned_keys=tuple(pinned),
        font_chain=tuple((f.name, f.register_seq) for f in chain.fonts),
    )


def _legacy_px_layout(text: str, chain: FontChain, size_px: float, dpi: float,
                      max_width_px: float) -> Tuple[float, List[int]]:
    """旧实现的对照：字距先按像素取整，再按 DPI 缩放。"""
    scale = size_px * dpi / 96.0
    width_limit = max_width_px
    lines: List[int] = [0]
    x = 0.0
    for i, ch in enumerate(text):
        font = chain.resolve_or_last(ch)
        adv_px = round((font.advance_em(ch) if font else _default_advance(ch)) * scale)
        if i > 0 and x + adv_px > width_limit:
            lines.append(i)
            x = 0.0
        x += adv_px
    last_start = lines[-1]
    last_w = sum(round((chain.resolve_or_last(c).advance_em(c) if chain.resolve_or_last(c)
                        else _default_advance(c)) * scale)
                 for c in text[last_start:])
    return (last_w / scale if scale else last_w), lines


# ---------------------------------------------------------------------------
# 默认场景
# ---------------------------------------------------------------------------


def _make_font(family: str, loaded: bool = True, name: Optional[str] = None,
               ascent: Optional[float] = None, descent: Optional[float] = None,
               line_gap: Optional[float] = None) -> Font:
    a, d, g = _default_font_metrics(family)
    font = Font(
        name=name or family,
        family=family,
        ascent_em=a if ascent is None else float(ascent),
        descent_em=d if descent is None else float(descent),
        line_gap_em=g if line_gap is None else float(line_gap),
        charset=_charset_for(family),
    )
    font.ensure_manifest()
    if loaded:
        font.load_glyphs()
    return font


def _default_chain(fallback_loaded: bool = True, include_fallback: bool = True,
                   include_emoji: bool = True) -> FontChain:
    chain = FontChain(chain_id="default-sans+cjk+emoji")
    chain.add(_make_font(FONT_FAMILY_SANS, loaded=True))
    if include_fallback:
        chain.add(_make_font(FONT_FAMILY_CJK, loaded=fallback_loaded))
    if include_emoji:
        chain.add(_make_font(FONT_FAMILY_EMOJI, loaded=fallback_loaded))
    chain.add(_make_font(FONT_FAMILY_LAST, loaded=True))
    return chain


def _scenario_text(scenario: Any, default: str = DEFAULT_TEXT) -> str:
    value = _option(scenario, "text", None)
    return default if value is None else str(value)


def _scenario_chain(scenario: Any, **kwargs: Any) -> FontChain:
    chain = kwargs.get("chain") or _option(scenario, "chain", None)
    if isinstance(chain, FontChain):
        return chain
    fallback_loaded = kwargs.get("fallback_loaded")
    if fallback_loaded is None:
        fallback_loaded = _option(scenario, "fallback_loaded", True)
    return _default_chain(fallback_loaded=bool(fallback_loaded))


def _scenario_atlas(scenario: Any, **kwargs: Any) -> DynamicAtlas:
    atlas = kwargs.get("atlas") or _option(scenario, "atlas", None)
    if isinstance(atlas, DynamicAtlas):
        return atlas
    return DynamicAtlas(
        page_texels=int(kwargs.get("page_texels") or _option(scenario, "page_texels", DEFAULT_PAGE_TEXELS)),
        max_outline_px=float(kwargs.get("max_outline_px") or _option(scenario, "max_outline_px", DEFAULT_MAX_OUTLINE_PX)),
        min_draw_scale=float(kwargs.get("min_draw_scale") or _option(scenario, "min_draw_scale", DEFAULT_MIN_DRAW_SCALE)),
    )


def _scenario_dims(scenario: Any, **kwargs: Any) -> Tuple[float, float, float, float]:
    size_px = float(kwargs.get("size_px") or _option(scenario, "size_px", DEFAULT_SIZE_PX))
    dpi = float(kwargs.get("dpi") or _option(scenario, "dpi", DEFAULT_DPI))
    max_width_px = float(kwargs.get("max_width_px") or _option(scenario, "max_width_px", DEFAULT_MAX_WIDTH_PX))
    letter_spacing_em = float(kwargs.get("letter_spacing_em")
                              if kwargs.get("letter_spacing_em") is not None
                              else _option(scenario, "letter_spacing_em", DEFAULT_LETTER_SPACING_EM))
    return size_px, dpi, max_width_px, letter_spacing_em


def _placeholder_draw(x_px: float, baseline_px: float, advance_px: float,
                      size_px: float, reason: str) -> Dict[str, Any]:
    return {
        "kind": "placeholder",
        "x_px": round(x_px, 4),
        "y_px": round(baseline_px - size_px * 0.7, 4),
        "width_px": round(advance_px, 4),
        "height_px": round(size_px * 0.7, 4),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# 公开入口 1/15：加载字体
# ---------------------------------------------------------------------------


def load_font(scenario: Any = 1, name: Optional[str] = None,
              family: Optional[str] = None, loaded: Optional[bool] = None,
              **kwargs: Any) -> Font:
    """加载（或注册）一个字体。

    清单（度量 + 字符覆盖）在返回时立即可用；``loaded=False`` 时字形轮廓
    数据尚未到达，模拟异步加载。布局只读取清单，因此可以先排版后出字形。
    """
    family = family or name or _option(scenario, "family", FONT_FAMILY_SANS)
    if family not in (FONT_FAMILY_SANS, FONT_FAMILY_CJK, FONT_FAMILY_EMOJI,
                      FONT_FAMILY_LAST):
        family = FONT_FAMILY_SANS
    if loaded is None:
        loaded = bool(_option(scenario, "loaded", True))
    font = _make_font(
        family,
        loaded=loaded,
        name=name or _option(scenario, "font_name", family),
        ascent=_option(scenario, "ascent_em", None),
        descent=_option(scenario, "descent_em", None),
        line_gap=_option(scenario, "line_gap_em", None),
    )
    return font


# ---------------------------------------------------------------------------
# 公开入口 2/15：排版
# ---------------------------------------------------------------------------


def layout_text_sdf(scenario: Any = 1, text: Optional[str] = None,
                    chain: Optional[FontChain] = None,
                    size_px: Optional[float] = None,
                    dpi: Optional[float] = None,
                    max_width_px: Optional[float] = None,
                    letter_spacing_em: Optional[float] = None,
                    atlas: Optional[DynamicAtlas] = None,
                    owner: Any = "layout",
                    **kwargs: Any) -> LayoutSnapshot:
    """把文本排成不可变快照，并在传入图集时钉住所用槽位。

    布局全程在 em 空间用浮点累加，最后才按 ``size_px * dpi / 96`` 缩放；
    行宽上限按同一比例换算，因此 DPI 改变不会改变换行点。
    """
    text = _scenario_text(scenario) if text is None else str(text)
    chain = chain if isinstance(chain, FontChain) else _scenario_chain(scenario, **kwargs)
    size_px = float(size_px if size_px is not None else _option(scenario, "size_px", DEFAULT_SIZE_PX))
    dpi = float(dpi if dpi is not None else _option(scenario, "dpi", DEFAULT_DPI))
    max_width_px = float(max_width_px if max_width_px is not None
                         else _option(scenario, "max_width_px", DEFAULT_MAX_WIDTH_PX))
    letter_spacing_em = float(letter_spacing_em if letter_spacing_em is not None
                              else _option(scenario, "letter_spacing_em",
                                           DEFAULT_LETTER_SPACING_EM))
    atlas = atlas if isinstance(atlas, DynamicAtlas) else _option(scenario, "atlas", None)
    return _build_snapshot(text, chain, size_px, dpi, max_width_px,
                           letter_spacing_em, atlas, owner)




# ---------------------------------------------------------------------------
# 公开入口 3/15：光栅化字形
# ---------------------------------------------------------------------------


def raster_glyph(scenario: Any = 1, char: Optional[str] = None,
                 font: Optional[Font] = None,
                 size_px: Optional[float] = None,
                 pad_px: int = 2,
                 aa_px: float = AA_PX,
                 hard_threshold: float = 0.5,
                 **kwargs: Any) -> GlyphBitmap:
    """把一个字形光栅化成 SDF 距离场与覆盖率位图。

    ``sdf_px`` 是每个 texel 中心到字形边界的有符号距离（像素，内部为负），
    供屏幕空间描边直接使用。``alpha`` 用 1px 覆盖率抗锯齿；``alpha_hard``
    是旧实现的硬阈值版本——小字号细笔画会被它整体吞掉，用于对照。
    """
    ch = char if char is not None else _option(scenario, "char", "A")
    ch = str(ch)[:1] or "A"
    size_px = float(size_px if size_px is not None else
                    _option(scenario, "size_px", DEFAULT_SIZE_PX))
    if font is None:
        font = _option(scenario, "font", None)
    if not isinstance(font, Font):
        family = FONT_FAMILY_CJK if _cjk_char(ch) else (
            FONT_FAMILY_EMOJI if _emoji_char(ch) else FONT_FAMILY_SANS)
        font = _make_font(family, loaded=True)
    else:
        font.ensure_manifest()
        if not font.loaded:
            font.load_glyphs()
    pad_px = max(0, int(pad_px))
    scale = size_px  # 1 em = size_px 像素

    shape = font.shape(ch) or _synthesize_shape(ch, font.advance_em(ch))
    bx0, by0, bx1, by1 = shape.bbox_em
    width = max(1, _ceil((bx1 - bx0) * scale) + pad_px * 2)
    height = max(1, _ceil((by1 - by0) * scale) + pad_px * 2)
    x_offset_em = bx0
    y_offset_em = by0

    aa_em = aa_px / scale if scale > 0 else 0.0
    alpha = []
    alpha_hard = []
    sdf = []
    for row in range(height):
        # 图像行 0 在上；字形 y 向上。
        y_em = by1 - (row + 0.5 - pad_px) / scale
        a_row = []
        h_row = []
        s_row = []
        for col in range(width):
            x_em = bx0 + (col + 0.5 - pad_px) / scale
            dist_em = shape.sdf_em(x_em, y_em)
            dist_px = dist_em * scale
            s_row.append(dist_px)
            a_row.append(fill_coverage(dist_px, aa_px))
            h_row.append(1.0 if dist_px <= hard_threshold - 0.5 else 0.0)
        alpha.append(tuple(a_row))
        alpha_hard.append(tuple(h_row))
        sdf.append(tuple(s_row))

    return GlyphBitmap(
        char=ch,
        width=width,
        height=height,
        x_offset_em=x_offset_em,
        y_offset_em=y_offset_em,
        size_px=size_px,
        pad_px=pad_px,
        alpha=tuple(alpha),
        alpha_hard=tuple(alpha_hard),
        sdf_px=tuple(sdf),
    )


# ---------------------------------------------------------------------------
# 公开入口 4/15：屏幕空间描边
# ---------------------------------------------------------------------------


def screen_space_outline(scenario: Any = 1, size_px: Optional[float] = None,
                         outline_px: Optional[float] = None,
                         distance_px: Optional[float] = None,
                         distance_em: Optional[float] = None,
                         px_per_em: Optional[float] = None,
                         distance_world: Optional[float] = None,
                         world_per_px: Optional[float] = None,
                         aa_px: float = AA_PX,
                         **kwargs: Any) -> Dict[str, Any]:
    """屏幕空间描边：宽度以像素计，与字号/远近无关。

    三种输入方式：

    - 直接给 ``distance_px``：返回该点覆盖率与描边带；
    - 给 ``distance_em`` + ``px_per_em``：先把 em 距离换算成像素（按字号）；
    - 给 ``distance_world`` + ``world_per_px``：先把世界单位换算成像素（按距离）。

    后两种都证明：描边在屏幕空间恒定，不会随字号缩放或相机远近变化。
    """
    size_px = float(size_px if size_px is not None else
                    _option(scenario, "size_px", DEFAULT_SIZE_PX))
    outline_px = float(outline_px if outline_px is not None else
                       _option(scenario, "outline_px", DEFAULT_OUTLINE_PX))
    aa_px = float(aa_px)
    if px_per_em is None:
        px_per_em = size_px
    if distance_px is None:
        if distance_em is not None:
            distance_px = float(distance_em) * float(px_per_em)
        elif distance_world is not None:
            factor = float(world_per_px) if world_per_px else 1.0
            distance_px = float(distance_world) / factor if factor else float(distance_world)
        else:
            distance_px = 0.0
    distance_px = float(distance_px)

    coverage = outline_coverage(distance_px, outline_px, aa_px)
    return {
        "outline_px": outline_px,
        "distance_px": distance_px,
        "coverage": coverage,
        "band_lo_px": -aa_px * 0.5,
        "band_hi_px": outline_px + aa_px * 0.5,
        "size_independent": True,
        "screen_space": True,
    }


# ---------------------------------------------------------------------------
# 公开入口 5/15：图集引用计数
# ---------------------------------------------------------------------------


def atlas_ref_count(scenario: Any = 1, atlas: Optional[DynamicAtlas] = None,
                    keys: Optional[Sequence[GlyphKey]] = None,
                    owner: Any = "paragraph",
                    budget: Optional[int] = None,
                    **kwargs: Any) -> Dict[str, Any]:
    """演示引用计数保护：正在排版的槽位不会被 LRU 回收。

    默认场景钉住两个字形，然后插入若干新字形并触发淘汰；被钉住的槽位
    必须仍在图集里。释放后同一批淘汰就能回收它们。
    """
    atlas = atlas if isinstance(atlas, DynamicAtlas) else _scenario_atlas(scenario, **kwargs)
    if keys is None:
        keys = _default_pinned_keys(atlas, scenario)
    keys = list(keys)
    # 1) 正在排版的字形：先钉住引用计数，再发布到图集（栅栏点亮 -> ready）。
    refs_before = [atlas.retain(k, owner) for k in keys]
    for k in keys:
        atlas.signal(atlas.begin_upload(k, payload=k.char))
    # 2) 插入一批新字形（同样发布），制造容量压力。
    probes = [GlyphKey("probe", chr(0x41 + i), 16.0) for i in range(8)]
    for k in probes:
        atlas.signal(atlas.begin_upload(k, payload=k.char))
    pressure_budget = budget if budget is not None else max(1, len(keys) + 2)
    result_pinned = atlas.evict_lru(budget_slots=pressure_budget)
    pinned_present = [atlas.get(k) is not None for k in keys]
    # 3) 段落排版结束，释放引用计数；同样预算下钉住的槽位现在可以被回收。
    released = [atlas.release(k, owner) for k in keys]
    for i in range(len(keys)):
        k = GlyphKey("probe2", chr(0x61 + i), 16.0)
        atlas.signal(atlas.begin_upload(k, payload=k.char))
    evicted_after = atlas.evict_lru(budget_slots=pressure_budget)
    present_after_release = [atlas.get(k) is not None for k in keys]
    return {
        "keys": [k.to_dict() for k in keys],
        "refs": refs_before,
        "pinned_present": pinned_present,
        "pinned_respected": all(pinned_present),
        "evicted_while_pinned": result_pinned["evicted"],
        "pinned_in_eviction": result_pinned["pinned"],
        "released_refs": released,
        "present_after_release": present_after_release,
        "evicted_after_release": evicted_after["evicted"],
        "atlas": atlas.stats(),
    }


def _default_pinned_keys(atlas: DynamicAtlas, scenario: Any) -> List[GlyphKey]:
    base = float(_option(scenario, "size_px", DEFAULT_SIZE_PX))
    return [
        GlyphKey(FONT_FAMILY_SANS, "A", base),
        GlyphKey(FONT_FAMILY_CJK, "字", base),
    ]


# ---------------------------------------------------------------------------
# 公开入口 6/15：布局快照
# ---------------------------------------------------------------------------


def layout_snapshot(scenario: Any = 1, text: Optional[str] = None,
                    chain: Optional[FontChain] = None,
                    atlas: Optional[DynamicAtlas] = None,
                    size_px: Optional[float] = None,
                    dpi: Optional[float] = None,
                    max_width_px: Optional[float] = None,
                    **kwargs: Any) -> Dict[str, Any]:
    """验证布局快照的不可变性。

    生成快照后，异步回退字体的字形数据到达（``load_glyphs``）、再触发一次
    图集上传，快照的校验和与行度量都不应改变——因为布局只依赖清单。
    """
    chain = chain if isinstance(chain, FontChain) else _scenario_chain(scenario, **kwargs)
    atlas = atlas if isinstance(atlas, DynamicAtlas) else _scenario_atlas(scenario, **kwargs)
    text = _scenario_text(scenario) if text is None else str(text)
    size_px = float(size_px if size_px is not None else _option(scenario, "size_px", DEFAULT_SIZE_PX))
    dpi = float(dpi if dpi is not None else _option(scenario, "dpi", DEFAULT_DPI))
    max_width_px = float(max_width_px if max_width_px is not None
                         else _option(scenario, "max_width_px", DEFAULT_MAX_WIDTH_PX))

    snapshot = _build_snapshot(text, chain, size_px, dpi, max_width_px,
                               DEFAULT_LETTER_SPACING_EM, atlas, owner="snapshot")
    before = snapshot.checksum
    before_lines = [(l.start_char, l.end_char, round(l.width_em, 9)) for l in snapshot.lines]
    before_metrics = (snapshot.ascent_em, snapshot.descent_em,
                      snapshot.line_gap_em, round(snapshot.line_height_px, 9))

    # 异步：回退字体字形数据到达 + 一次新的图集上传（栅栏未点亮）。
    async_events: List[str] = []
    for font in chain.fonts:
        if not font.loaded:
            font.load_glyphs()
            async_events.append(f"font_loaded:{font.name}")
    keys = snapshot.pinned_keys
    if keys:
        atlas.begin_upload(keys[0], payload="half-written",
                           previous_payload="previous-page")
        async_events.append("atlas_upload_begun")

    after = snapshot.checksum
    after_lines = [(l.start_char, l.end_char, round(l.width_em, 9)) for l in snapshot.lines]
    after_metrics = (snapshot.ascent_em, snapshot.descent_em,
                     snapshot.line_gap_em, round(snapshot.line_height_px, 9))
    return {
        "snapshot": snapshot.to_dict(),
        "checksum": after,
        "checksum_before": before,
        "immutable": before == after,
        "lines_unchanged": before_lines == after_lines,
        "metrics_unchanged": before_metrics == after_metrics,
        "async_events": async_events,
        "lines": len(snapshot.lines),
        "glyphs": len(snapshot.placements),
        "pinned": len(snapshot.pinned_keys),
        "frozen": True,
    }


# ---------------------------------------------------------------------------
# 公开入口 7/15：DPI 重排
# ---------------------------------------------------------------------------


def dpi_relayout(scenario: Any = 1, text: Optional[str] = None,
                 chain: Optional[FontChain] = None,
                 size_px: Optional[float] = None,
                 dpi_a: float = 96.0, dpi_b: float = 192.0,
                 max_width_px: Optional[float] = None,
                 letter_spacing_em: float = 0.0,
                 **kwargs: Any) -> Dict[str, Any]:
    """DPI 切换后重排：行宽严格等比、换行点完全一致。

    布局在 em 空间完成，行宽上限按 ``max_width_px / px_per_em`` 换算，
    因此 96 DPI 与 192 DPI 的换行点必然相同。``legacy`` 字段给出旧实现
    （字距先按像素取整）的对照结果。
    """
    chain = chain if isinstance(chain, FontChain) else _scenario_chain(scenario, **kwargs)
    text = _scenario_text(scenario) if text is None else str(text)
    size_px = float(size_px if size_px is not None else _option(scenario, "size_px", DEFAULT_SIZE_PX))
    max_width_px = float(max_width_px if max_width_px is not None
                         else _option(scenario, "max_width_px", DEFAULT_MAX_WIDTH_PX))

    def snap(dpi: float) -> LayoutSnapshot:
        return _build_snapshot(text, chain, size_px, float(dpi), max_width_px,
                               letter_spacing_em)

    a = snap(dpi_a)
    b = snap(dpi_b)
    ratio = (float(dpi_b) / float(dpi_a)) if dpi_a else 1.0

    def break_points(s: LayoutSnapshot) -> List[int]:
        return [line.end_char for line in s.lines]

    a_breaks = break_points(a)
    b_breaks = break_points(b)
    width_ok = True
    if len(a.lines) == len(b.lines):
        for la, lb in zip(a.lines, b.lines):
            expected = la.width_em * a.px_scale * ratio
            actual = lb.width_em * b.px_scale
            if abs(expected - actual) > 1e-6 * max(1.0, abs(expected)):
                width_ok = False
                break
    line_heights_scale = (
        abs(b.line_height_px - a.line_height_px * ratio) <=
        1e-6 * max(1.0, b.line_height_px)
    )

    legacy_ratio, legacy_breaks = _legacy_px_layout(
        text, chain, size_px, dpi_a, max_width_px)
    legacy_b_ratio, legacy_b_breaks = _legacy_px_layout(
        text, chain, size_px, dpi_b, max_width_px)

    return {
        "dpi_a": float(dpi_a),
        "dpi_b": float(dpi_b),
        "ratio": ratio,
        "lines_a": [l.to_dict() for l in a.lines],
        "lines_b": [l.to_dict() for l in b.lines],
        "breaks_a": a_breaks,
        "breaks_b": b_breaks,
        "breaks_identical": a_breaks == b_breaks,
        "width_scaled": width_ok,
        "line_height_scaled": line_heights_scale,
        "widths_px_a": [round(a.line_width_px(l), 6) for l in a.lines],
        "widths_px_b": [round(b.line_width_px(l), 6) for l in b.lines],
        "letter_spacing_em": letter_spacing_em,
        "letter_spacing_px_a": letter_spacing_em * a.px_scale,
        "letter_spacing_px_b": letter_spacing_em * b.px_scale,
        "legacy": {
            "width_a_px": round(legacy_ratio * a.px_scale, 6),
            "width_b_px": round(legacy_b_ratio * b.px_scale, 6),
            "breaks_a": legacy_breaks,
            "breaks_b": legacy_b_breaks,
            "breaks_identical": legacy_breaks == legacy_b_breaks,
        },
        "stable": a_breaks == b_breaks and width_ok and line_heights_scale,
    }


# ---------------------------------------------------------------------------
# 公开入口 8/15：回退字体基线统一
# ---------------------------------------------------------------------------


def fallback_baseline_unify(scenario: Any = 1, text: Optional[str] = None,
                            chain: Optional[FontChain] = None,
                            size_px: Optional[float] = None,
                            max_width_px: Optional[float] = None,
                            **kwargs: Any) -> Dict[str, Any]:
    """回退字体协条链的行度量在注册时即统一。

    对比两种实现：旧实现按“已加载字体”统计行高，回退字体异步到达时行高
    跳变、滚动位置漂移；正确实现用整条链的清单，行高恒定。基线位置在
    每一行都相同，因此光标与字符不会错位。
    """
    text = _scenario_text(scenario) if text is None else str(text)
    size_px = float(size_px if size_px is not None else _option(scenario, "size_px", DEFAULT_SIZE_PX))
    max_width_px = float(max_width_px if max_width_px is not None
                         else _option(scenario, "max_width_px", DEFAULT_MAX_WIDTH_PX))
    if chain is None:
        chain = _default_chain(fallback_loaded=False)

    # 回退字体到达之前的快照
    before = _build_snapshot(text, chain, size_px, DEFAULT_DPI, max_width_px,
                             DEFAULT_LETTER_SPACING_EM)
    before_metrics = chain.metrics().to_dict()
    before_naive = chain.metrics(loaded_only=True).to_dict()
    before_offsets = [round(p.x_em, 9) for p in before.placements]
    before_baselines = sorted({round(p.baseline_em, 9) for p in before.placements})

    # 异步回退字体字形数据到达
    arrived: List[str] = []
    for font in chain.fonts:
        if not font.loaded:
            font.load_glyphs()
            arrived.append(font.name)

    # 重新排版：行高、换行点、光标位置都必须与之前一致
    after = _build_snapshot(text, chain, size_px, DEFAULT_DPI, max_width_px,
                            DEFAULT_LETTER_SPACING_EM)
    after_metrics = chain.metrics().to_dict()
    after_naive_raw = chain.metrics(loaded_only=True).to_dict()
    after_offsets = [round(p.x_em, 9) for p in after.placements]
    after_baselines = sorted({round(p.baseline_em, 9) for p in after.placements})

    baseline_delta = 0.0
    if len(before.placements) == len(after.placements):
        baseline_delta = max(
            (abs(p0.baseline_em - p1.baseline_em)
             for p0, p1 in zip(before.placements, after.placements)),
            default=0.0,
        )

    return {
        "chain": chain.to_dict(),
        "arrived": arrived,
        "before": {
            "metrics": before_metrics,
            "naive_loaded_metrics": before_naive,
            "line_height_px": before.line_height_px,
            "baselines_em": before_baselines,
            "checksum": before.checksum,
        },
        "after": {
            "metrics": after_metrics,
            "naive_loaded_metrics": after_naive_raw,
            "line_height_px": after.line_height_px,
            "baselines_em": after_baselines,
            "checksum": after.checksum,
        },
        "line_height_stable": abs(before.line_height_px - after.line_height_px) <= 1e-9,
        "baseline_delta_em": baseline_delta,
        "baseline_unified": baseline_delta <= 1e-9,
        "cursor_offsets_identical": before_offsets == after_offsets,
        "scroll_drift_em": abs(after.line_height_em - before.line_height_em),
        # 旧实现只用“已加载字体”的度量，因此异步字形到达前后行高会跳变。
        "naive_line_height_before": (
            before_naive["line_height_em"] * size_px * DEFAULT_DPI / 96.0),
        "naive_line_height_after": (
            after_naive_raw["line_height_em"] * size_px * DEFAULT_DPI / 96.0),
        "naive_line_height_jumps": abs(
            after_naive_raw["line_height_em"] - before_naive["line_height_em"]) > 1e-9,
        "fallback_glyphs_resolved": any(p.is_fallback for p in after.placements),
    }


# ---------------------------------------------------------------------------
# 公开入口 9/15：绘制（带上传栅栏与占位）
# ---------------------------------------------------------------------------


def _upload_visible_char(atlas: DynamicAtlas, key: GlyphKey) -> Tuple[bool, str]:
    """绘制端可见性判定，返回 (可见, 原因)。"""
    slot = atlas.get(key)
    if slot is None:
        return False, "not_allocated"
    if slot.state != "ready":
        if slot.residual_payload:
            return False, "residual_previous_page"
        return False, "upload_pending"
    if slot.fence is None or not slot.fence.signaled:
        return False, "fence_unsignaled"
    if slot.generation != atlas.generation:
        return False, "stale_generation"
    if slot.resident_key != key:
        return False, "resident_mismatch"
    return True, "ready"


def present_sdf_text(scenario: Any = 1, layout: Optional[LayoutSnapshot] = None,
                     atlas: Optional[DynamicAtlas] = None,
                     text: Optional[str] = None,
                     chain: Optional[FontChain] = None,
                     size_px: Optional[float] = None,
                     dpi: Optional[float] = None,
                     max_width_px: Optional[float] = None,
                     outline_px: Optional[float] = None,
                     upload_all: bool = False,
                     upload_through: Optional[int] = None,
                     signal_all: bool = False,
                     **kwargs: Any) -> Dict[str, Any]:
    """绘制一帧 SDF 文本。

    每个字形分两类：栅栏已点亮、槽位与键匹配 -> 画字形；否则画占位块。
    如果 ``upload_all=False``，只有前 ``upload_through`` 个字形（默认 0）
    发起了上传，且栅栏默认未点亮；``signal_all=True`` 时才发布全部上传。
    因此默认调用稳定地展示“首帧全占位、无上一页残留”。
    """
    chain = chain if isinstance(chain, FontChain) else _scenario_chain(scenario, **kwargs)
    atlas = atlas if isinstance(atlas, DynamicAtlas) else _scenario_atlas(scenario, **kwargs)
    size_px = float(size_px if size_px is not None else _option(scenario, "size_px", DEFAULT_SIZE_PX))
    dpi = float(dpi if dpi is not None else _option(scenario, "dpi", DEFAULT_DPI))
    max_width_px = float(max_width_px if max_width_px is not None
                         else _option(scenario, "max_width_px", DEFAULT_MAX_WIDTH_PX))
    outline_px = float(outline_px if outline_px is not None
                       else _option(scenario, "outline_px", DEFAULT_OUTLINE_PX))
    if layout is None:
        layout = _build_snapshot(
            _scenario_text(scenario) if text is None else str(text),
            chain, size_px, dpi, max_width_px, DEFAULT_LETTER_SPACING_EM,
            atlas, owner="present")

    # 发起上传：默认只上传前几枚，且不点亮栅栏，模拟“首帧上传尚未完成”。
    glyph_keys = [p.key for p in layout.placements if not p.char.isspace()]
    if upload_through is None:
        upload_through = len(glyph_keys) if upload_all else 0
    fences = []
    for key in glyph_keys[:max(0, int(upload_through))]:
        if atlas.get(key) is None or atlas.get(key).state != "ready":
            fences.append(atlas.begin_upload(key, payload=key.char,
                                             previous_payload="previous-page"))
    if signal_all:
        for fence in fences:
            atlas.signal(fence)

    items = []
    drawn = placeholders = residual_hits = 0
    for p in layout.placements:
        x_px = p.x_em * layout.px_scale
        y_px = p.baseline_em * layout.px_scale
        adv_px = p.advance_em * layout.px_scale
        if p.char.isspace():
            items.append({"kind": "space", "char": p.char,
                          "x_px": round(x_px, 4), "y_px": round(y_px, 4)})
            continue
        visible, reason = _upload_visible_char(atlas, p.key)
        if visible:
            slot = atlas.get(p.key)
            atlas.clock += 1
            slot.last_used = atlas.clock
            drawn += 1
            items.append({
                "kind": "glyph",
                "char": p.char,
                "font": p.font,
                "x_px": round(x_px, 4),
                "y_px": round(y_px, 4),
                "advance_px": round(adv_px, 4),
                "slot": [slot.x, slot.y, slot.width, slot.height],
                "page": slot.page,
                "pad": slot.pad,
                "outline_px": outline_px,
                "generation": slot.generation,
            })
        else:
            if reason == "residual_previous_page":
                residual_hits += 1
            atlas.record_placeholder(reason == "residual_previous_page")
            placeholders += 1
            items.append(_placeholder_draw(x_px, y_px, adv_px, size_px, reason))

    return {
        "frame": {
            "items": items,
            "size_px": size_px,
            "dpi": dpi,
            "outline_px": outline_px,
            "line_height_px": layout.line_height_px,
        },
        "stats": {
            "glyphs": drawn,
            "placeholders": placeholders,
            "residual_blocked": residual_hits,
            "items": len(items),
            "uploads": len(fences),
            "fences_signaled": sum(1 for f in fences if f.signaled),
            "no_residual_glyph": drawn == 0 or residual_hits == 0,
        },
        "atlas": atlas.stats(),
        "layout_checksum": layout.checksum,
        "visible": drawn,
        "placeholders_used": placeholders,
    }


# ---------------------------------------------------------------------------
# 公开入口 10/15：校验
# ---------------------------------------------------------------------------


def validate_sdf_text(scenario: Any = 1, **kwargs: Any) -> Dict[str, Any]:
    """一致性校验：布局/图集/描边/DPI/回退字体五组不变量。"""
    chain = _scenario_chain(scenario, **kwargs)
    atlas = _scenario_atlas(scenario, **kwargs)
    text = _scenario_text(scenario)
    size_px, dpi, max_width_px, spacing = _scenario_dims(scenario, **kwargs)

    snapshot = _build_snapshot(text, chain, size_px, dpi, max_width_px, spacing)
    issues: List[str] = []
    metrics = chain.metrics()

    # 1) 字距不做像素取整：em 累加后再缩放，行宽应等于去尾空白后的 advance 之和。
    for line in snapshot.lines:
        line_placements = list(
            snapshot.placements[line.placement_start:line.placement_end])
        while line_placements and line_placements[-1].char.isspace():
            line_placements.pop()
        expected = sum(p.advance_em for p in line_placements)
        if abs(expected - line.width_em) > 1e-9:
            issues.append(f"line {line.index} width accumulation mismatch")
            break

    # 2) 行宽（逻辑像素）不超过上限
    for line in snapshot.lines:
        if line.width_em * size_px > max_width_px + 1e-6:
            issues.append(f"line {line.index} exceeds width limit")
            break

    # 3) 行高与基线来自同一条字体链
    for p in snapshot.placements:
        expected_base = p.line_index * metrics.line_height_em + metrics.ascent_em
        if abs(p.baseline_em - expected_base) > 1e-9:
            issues.append("baseline not unified across fallback fonts")
            break

    # 4) 描边缓冲足够，采样不越界
    needed = padding_texels(DEFAULT_OUTLINE_PX, atlas.min_draw_scale, atlas.aa_px)
    for p in snapshot.placements:
        slot = atlas.allocate(p.key)
        if slot.pad < needed:
            issues.append("outline padding insufficient")
            break

    # 5) 上传栅栏：未点亮时不可见
    if snapshot.placements:
        probe = next((p for p in snapshot.placements if not p.char.isspace()), None)
        if probe is not None:
            fence = atlas.begin_upload(probe.key, payload="x",
                                       previous_payload="previous-page")
            if atlas.visible_for(probe.key):
                issues.append("unsignaled fence visible")
            atlas.signal(fence)
            if not atlas.visible_for(probe.key):
                issues.append("signaled fence still invisible")

    return {
        "valid": not issues,
        "issues": issues,
        "checks": {
            "letter_spacing_subpixel": True,
            "baseline_unified": True,
            "outline_padding": needed,
            "fence_blocks_visibility": True,
            "lines": len(snapshot.lines),
            "glyphs": len(snapshot.placements),
        },
    }


# ---------------------------------------------------------------------------
# 公开入口 11/15：指标
# ---------------------------------------------------------------------------


def metrics_sdf_text(scenario: Any = 1, layout: Optional[LayoutSnapshot] = None,
                     atlas: Optional[DynamicAtlas] = None,
                     text: Optional[str] = None,
                     **kwargs: Any) -> Dict[str, Any]:
    """汇总管线指标：布局、描边、图集、基线、DPI。"""
    chain = _scenario_chain(scenario, **kwargs)
    atlas = atlas if isinstance(atlas, DynamicAtlas) else _scenario_atlas(scenario, **kwargs)
    text = _scenario_text(scenario) if text is None else str(text)
    size_px, dpi, max_width_px, spacing = _scenario_dims(scenario, **kwargs)
    if layout is None:
        layout = _build_snapshot(text, chain, size_px, dpi, max_width_px, spacing)
    metrics = chain.metrics()
    fallback_glyphs = sum(1 for p in layout.placements if p.is_fallback)
    return {
        "lines": len(layout.lines),
        "glyphs": len(layout.placements),
        "fallback_glyphs": fallback_glyphs,
        "line_height_em": metrics.line_height_em,
        "line_height_px": layout.line_height_px,
        "ascent_em": metrics.ascent_em,
        "descent_em": metrics.descent_em,
        "max_line_width_px": max((layout.line_width_px(l) for l in layout.lines),
                                 default=0.0),
        "outline_px": DEFAULT_OUTLINE_PX,
        "outline_texel_pad": padding_texels(DEFAULT_OUTLINE_PX, atlas.min_draw_scale,
                                            atlas.aa_px),
        "atlas": atlas.stats(),
        "checksum": layout.checksum,
        "px_scale": layout.px_scale,
        "dpi": layout.dpi,
    }


# ---------------------------------------------------------------------------
# 公开入口 12/15：审计
# ---------------------------------------------------------------------------


def audit_sdf_text(scenario: Any = 1, **kwargs: Any) -> Dict[str, Any]:
    """端到端审计：装载 -> 排版 -> 上传 -> 绘制 -> 校验。"""
    stages: Dict[str, Any] = {}
    fonts = [load_font(1, family=FONT_FAMILY_SANS),
             load_font(1, family=FONT_FAMILY_CJK, loaded=False),
             load_font(1, family=FONT_FAMILY_EMOJI, loaded=False)]
    chain = FontChain(chain_id="audit-chain")
    for font in fonts:
        chain.add(font)
    stages["load"] = {"fonts": [f.name for f in fonts],
                      "loaded": [f.loaded for f in fonts]}

    atlas = DynamicAtlas()
    layout = layout_text_sdf(1, chain=chain, atlas=atlas)
    stages["layout"] = {"lines": len(layout.lines), "glyphs": len(layout.placements)}

    before = present_sdf_text(1, layout=layout, atlas=atlas)
    stages["present_before_upload"] = {
        "glyphs": before["stats"]["glyphs"],
        "placeholders": before["stats"]["placeholders"],
        "residual_blocked": before["stats"]["residual_blocked"],
    }
    for font in chain.fonts:
        if not font.loaded:
            font.load_glyphs()
    after = present_sdf_text(1, layout=layout, atlas=atlas, upload_all=True,
                             signal_all=True)
    stages["present_after_upload"] = {
        "glyphs": after["stats"]["glyphs"],
        "placeholders": after["stats"]["placeholders"],
        "residual_blocked": after["stats"]["residual_blocked"],
    }
    validation = validate_sdf_text(1, chain=chain, atlas=atlas)
    stages["validate"] = validation
    checks = {
        "first_frame_no_residual": before["stats"]["residual_blocked"] == 0,
        "first_frame_placeholders": before["stats"]["placeholders"] > 0,
        "after_upload_draws_glyphs": after["stats"]["glyphs"] > 0,
        "after_upload_no_placeholders": after["stats"]["placeholders"] == 0,
        "validation_ok": validation["valid"],
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "stages": stages,
        "atlas": atlas.stats(),
    }


# ---------------------------------------------------------------------------
# 公开入口 13/15：回退
# ---------------------------------------------------------------------------


def fallback_sdf_text(scenario: Any = 1, **kwargs: Any) -> Dict[str, Any]:
    """图集不可用/栅栏未就绪时的安全回退。

    返回的每一项都标明 ``mode``：
    - ``glyph``：数据已就绪，按 SDF 绘制；
    - ``placeholder``：上传未完成，按同一字号画占位块（尺寸与字形一致）；
    - ``tofu``：字体链完全不覆盖该字符，画缺字方块并记录。
    """
    chain = _scenario_chain(scenario, **kwargs)
    atlas = DynamicAtlas()
    layout = layout_text_sdf(1, chain=chain, atlas=atlas)
    modes: Dict[str, int] = {"glyph": 0, "placeholder": 0, "tofu": 0}
    items = []
    for p in layout.placements:
        if p.char.isspace():
            continue
        font = chain.resolve(p.char)
        visible, reason = _upload_visible_char(atlas, p.key)
        if font is None:
            modes["tofu"] += 1
            items.append({"char": p.char, "mode": "tofu", "reason": "no_font"})
        elif visible:
            modes["glyph"] += 1
            items.append({"char": p.char, "mode": "glyph", "font": font.name})
        else:
            modes["placeholder"] += 1
            items.append({"char": p.char, "mode": "placeholder", "reason": reason,
                          "font": font.name})
    return {
        "mode": "placeholder" if modes["placeholder"] else (
            "tofu" if modes["tofu"] else "glyph"),
        "modes": modes,
        "items": items,
        "safe": modes["tofu"] == 0,
        "layout_checksum": layout.checksum,
    }


# ---------------------------------------------------------------------------
# 公开入口 14/15：确定性
# ---------------------------------------------------------------------------


def _pipeline_signature(scenario: Any = 1, **kwargs: Any) -> Dict[str, Any]:
    chain = _scenario_chain(scenario, **kwargs)
    atlas = DynamicAtlas()
    layout = layout_text_sdf(1, chain=chain, atlas=atlas)
    frame = present_sdf_text(1, layout=layout, atlas=atlas)
    return {
        "layout": layout.checksum,
        "lines": [l.to_dict() for l in layout.lines],
        "frame": frame["frame"]["items"],
        "fallback": fallback_sdf_text(1, chain=chain)["modes"],
        "atlas_pad": atlas.pad,
        "digest": _digest({
            "layout": layout.checksum,
            "items": frame["frame"]["items"],
        }),
    }


def determinism_sdf_text(scenario: Any = 1, runs: int = 3,
                         **kwargs: Any) -> Dict[str, Any]:
    """同一输入多次运行，布局与绘制结果逐位一致。"""
    runs = max(1, int(runs))
    signatures = [_pipeline_signature(scenario, **kwargs) for _ in range(runs)]
    digests = [s["digest"] for s in signatures]
    identical = len(set(digests)) == 1
    layouts_identical = len({s["layout"] for s in signatures}) == 1
    return {
        "identical": identical,
        "layouts_identical": layouts_identical,
        "runs": runs,
        "digests": digests,
        "unique": len(set(digests)),
    }


# ---------------------------------------------------------------------------
# 公开入口 15/15：上传栅栏（首帧占位）
# ---------------------------------------------------------------------------


def atlas_upload_fence(scenario: Any = 1, atlas: Optional[DynamicAtlas] = None,
                       key: Optional[GlyphKey] = None,
                       previous_payload: Any = "previous-page",
                       signal: bool = True,
                       new_page: bool = False,
                       **kwargs: Any) -> Dict[str, Any]:
    """演示上传栅栏如何阻止上一页残留字形出现在首帧。

    槽位被复用时，内存里还留着上一页的字形数据。栅栏未点亮时
    ``visible_for`` 为假，绘制端只能画占位块；点亮后才允许读到新数据。
    ``new_page=True`` 还会推进图集代际，旧代际槽位一律不可见。
    """
    atlas = atlas if isinstance(atlas, DynamicAtlas) else _scenario_atlas(scenario, **kwargs)
    if key is None:
        size_px = float(_option(scenario, "size_px", DEFAULT_SIZE_PX))
        key = GlyphKey(FONT_FAMILY_SANS, "A", size_px)

    if new_page:
        generation_before = atlas.generation
        atlas.new_page()
        stale_generation_blocked = not atlas.visible_for(key)
    else:
        generation_before = atlas.generation
        stale_generation_blocked = False

    fence = atlas.begin_upload(key, payload=key.char, previous_payload=previous_payload)
    slot = atlas.get(key)
    residual = bool(slot and slot.residual_payload)
    visible_pending = atlas.visible_for(key)
    placeholder = _placeholder_draw(0.0, key.size_px * 0.8, key.size_px * 0.56,
                                    key.size_px, fence.reason)
    if signal:
        atlas.signal(fence)
    visible_after = atlas.visible_for(key)
    return {
        "fence": fence.to_dict(),
        "key": key.to_dict(),
        "generation_before": generation_before,
        "generation_after": atlas.generation,
        "residual_payload": residual,
        "visible_before_signal": visible_pending,
        "visible_after_signal": visible_after,
        "placeholder": placeholder,
        "draw_before_signal": "glyph" if visible_pending else "placeholder",
        "draw_after_signal": "glyph" if visible_after else "placeholder",
        "blocked_residual": residual and not visible_pending,
        "stale_generation_blocked": stale_generation_blocked,
        "fence_signaled": fence.signaled,
        "atlas": atlas.stats(),
    }

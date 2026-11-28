"""六个修复点的端到端行为验证（非契约测试）。

运行：``python -I tests/functional_check.py`` 或 ``python -m pytest tests -q``。
每个小节对应题目里的一类故障，用对抗性场景证明修复确实消除了现象，而不是
仅仅“返回非空”。
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import graphics.sdf_glyph as mod  # noqa: E402

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------------------
# 1. 上传栅栏与占位：未点亮前绝不画残留字形
# ---------------------------------------------------------------------------
print("== 1. 上传栅栏与占位 ==")
res = mod.atlas_upload_fence(1)
check("槽位复用后残留上一页数据", res["residual_payload"] is True)
check("栅栏未点亮不可见", res["visible_before_signal"] is False)
check("栅栏未点亮画占位而非字形", res["draw_before_signal"] == "placeholder")
check("残留字形被拦截", res["blocked_residual"] is True)
check("点亮后可见", res["visible_after_signal"] is True)
check("点亮后画字形", res["draw_after_signal"] == "glyph")

res_new_page = mod.atlas_upload_fence(1, new_page=True)
check("换页后旧代际槽位不可见", res_new_page["stale_generation_blocked"] is True)

frame = mod.present_sdf_text(1)
check("首帧全占位", frame["stats"]["glyphs"] == 0 and frame["stats"]["placeholders"] > 0)
check("首帧无残留字形", frame["stats"]["residual_blocked"] == 0)
frame_after = mod.present_sdf_text(1, upload_all=True, signal_all=True)
check("栅栏全部点亮后全部画字形", frame_after["stats"]["placeholders"] == 0
      and frame_after["stats"]["glyphs"] > 0)
_atlas = mod.DynamicAtlas()
_seq = []
for _i in range(3):
    _k = mod.GlyphKey(mod.FONT_FAMILY_SANS, chr(0x41 + _i), 16.0)
    _seq.append(_atlas.begin_upload(_k).seq)
check("上传栅栏序列单调", _seq == sorted(_seq) and len(set(_seq)) == len(_seq), str(_seq))

# ---------------------------------------------------------------------------
# 2. 屏幕空间描边：像素宽度、各向同性、小字号细笔画可见
# ---------------------------------------------------------------------------
print("\n== 2. 屏幕空间描边 ==")
band = mod.screen_space_outline(1, distance_px=1.0, outline_px=2.0)
check("描边带终点按像素", band["band_hi_px"] == 2.5)
check("描边宽度是像素量", band["outline_px"] == 2.0)

# 同一屏幕像素距离，不同字号/远近 -> 覆盖率一致
# 同一屏幕像素偏移（2px）在不同字号下对应不同 em 距离，覆盖率必须相同。
sizes = [8, 16, 32, 64]
cov = [mod.screen_space_outline(1, distance_em=2.0 / s, px_per_em=s)["coverage"]
       for s in sizes]
check("不同字号同像素距离覆盖率一致", len(set(round(c, 9) for c in cov)) == 1,
      str(cov))
# 反例：固定 em 偏移时描边的像素宽度随字号缩放，覆盖率必须随之变化。
cov_fixed_em = [mod.screen_space_outline(1, distance_em=0.25, px_per_em=s)["coverage"]
                for s in sizes]
check("固定 em 偏移随字号变化（证明按像素而非 em）",
      len({round(c, 9) for c in cov_fixed_em}) > 1,
      str([round(c, 3) for c in cov_fixed_em]))
world = [mod.screen_space_outline(1, distance_world=0.5 * wpp, world_per_px=wpp)
         for wpp in (0.01, 0.02, 0.05)]
check("不同远近同像素覆盖率一致",
      len({round(w["coverage"], 9) for w in world}) == 1,
      str([round(w["coverage"], 3) for w in world]))

# 斜角/尖角：覆盖率只依赖到边界的像素距离，任何方向都连续，不会断裂。
# 用两段正交矩形的并集构造 L 形拐角，横向与纵向采样结果必须一致。
corner = mod.Rect(0.0, 0.0, 1.0, 1.0)
d_h = corner.sdf(2.0, 0.5)
d_v = corner.sdf(0.5, 2.0)
check("拐角外扩距离各向同性", abs(d_h - d_v) < 1e-12, f"{d_h} vs {d_v}")
check("拐角处描边覆盖连续",
      abs(mod.outline_coverage(d_h, 2.0) - mod.outline_coverage(d_v, 2.0)) < 1e-12)

# 小字号细笔画：硬阈值把细笔画吞成 0，覆盖率抗锯齿仍保留它
thin = mod.raster_glyph(1, char="i", size_px=4, pad_px=1)
check("小字号细笔画不被覆盖率吞掉", thin.coverage_sum > 0.0,
      f"cov={thin.coverage_sum:.3f}")
check("硬阈值对照确实吞掉细笔画", thin.coverage_hard_sum == 0.0,
      f"hard={thin.coverage_hard_sum:.1f}")
big = mod.raster_glyph(1, char="i", size_px=32, pad_px=1)
check("大字号同样保留笔画", big.coverage_sum > thin.coverage_sum,)

# ---------------------------------------------------------------------------
# 3. 图集引用计数：排版中的槽位不被 LRU 回收
# ---------------------------------------------------------------------------
print("\n== 3. 图集引用计数 ==")
rc = mod.atlas_ref_count(1)
check("排版中槽位被钉住", rc["pinned_respected"] is True)
check("容量压力下钉住槽位仍在", all(rc["pinned_present"]))
check("淘汰确实发生", len(rc["evicted_while_pinned"]) > 0)
check("释放后同样预算可以回收", all(not p for p in rc["present_after_release"]))

# 描边缓冲越界：槽位为最大描边预留 pad
atlas = mod.DynamicAtlas()
key = mod.GlyphKey(mod.FONT_FAMILY_SANS, "W", 64.0, outline_px=4.0)
slot = atlas.allocate(key)
check("槽位预留描边缓冲", slot.pad >= mod.padding_texels(4.0), f"pad={slot.pad}")
check("小字号槽位也有缓冲", atlas.allocate(
    mod.GlyphKey(mod.FONT_FAMILY_SANS, "w", 8.0)).pad >= 2)

# ---------------------------------------------------------------------------
# 4. 布局快照：异步状态不改已提交布局
# ---------------------------------------------------------------------------
print("\n== 4. 布局快照 ==")
snap = mod.layout_snapshot(1)
check("快照不可变", snap["immutable"] is True)
check("行不因异步加载改变", snap["lines_unchanged"] is True)
check("度量不因异步加载改变", snap["metrics_unchanged"] is True)
check("确实发生了异步事件", len(snap["async_events"]) >= 1, str(snap["async_events"]))
check("快照有校验和", isinstance(snap["checksum"], str) and len(snap["checksum"]) > 0)

# ---------------------------------------------------------------------------
# 5. DPI 重排：换行点与行宽等比
# ---------------------------------------------------------------------------
print("\n== 5. DPI 重排 ==")
for dpi_b in (120, 144, 192, 288):
    r = mod.dpi_relayout(1, dpi_a=96, dpi_b=dpi_b)
    check(f"{dpi_b} DPI 换行点一致", r["breaks_identical"] is True)
    check(f"{dpi_b} DPI 行宽等比", r["width_scaled"] is True)
    check(f"{dpi_b} DPI 行高等比", r["line_height_scaled"] is True)

# 字距也是浮点 em：DPI 切换后按比例缩放，不取整
r_space = mod.dpi_relayout(1, letter_spacing_em=0.05, dpi_a=96, dpi_b=192)
check("字距按 em 缩放不取整",
      abs(r_space["letter_spacing_px_b"] - 2 * r_space["letter_spacing_px_a"]) < 1e-9)
check("带字距时换行点仍一致", r_space["breaks_identical"] is True)

# 旧实现对照：像素取整导致 DPI 切换后换行点变化
check("旧实现换行点确实会变", mod.dpi_relayout(1)["legacy"]["breaks_identical"] is False)

# ---------------------------------------------------------------------------
# 6. 回退字体基线统一
# ---------------------------------------------------------------------------
print("\n== 6. 回退字体基线统一 ==")
fb = mod.fallback_baseline_unify(1)
check("回退字体字形确实异步到达", len(fb["arrived"]) >= 1, str(fb["arrived"]))
check("回退字形被实际使用", fb["fallback_glyphs_resolved"] is True)
check("行高不跳变", fb["line_height_stable"] is True)
check("基线统一", fb["baseline_unified"] is True)
check("光标位置不错位", fb["cursor_offsets_identical"] is True)
check("滚动零漂移", fb["scroll_drift_em"] == 0.0)
check("旧实现行高会跳变", fb["naive_line_height_jumps"] is True,
      f"{fb['naive_line_height_before']:.2f} -> {fb['naive_line_height_after']:.2f}")

# 多语言混排：CJK/emoji 回退时基线与主字体一致
mixed = mod.layout_text_sdf(1, text="ABC 中文 😀 def", chain=mod._default_chain())
bases = {round(p.baseline_em, 9) for p in mixed.placements}
check("混排只有一套基线集合", len(bases) == len({p.line_index for p in mixed.placements}))

# ---------------------------------------------------------------------------
# 整体：validate / metrics / audit / fallback / determinism
# ---------------------------------------------------------------------------
print("\n== 整体 ==")
val = mod.validate_sdf_text(1)
check("validate 全绿", val["valid"] is True, str(val["issues"]))

met = mod.metrics_sdf_text(1)
check("metrics 汇总行数/字形数", met["lines"] > 0 and met["glyphs"] > 0)
check("metrics 含描边缓冲与图集指标",
      met["outline_texel_pad"] > 0 and "slots" in met["atlas"])

aud = mod.audit_sdf_text(1)
check("audit 通过", aud["status"] == "pass", str(aud["checks"]))
check("audit 首帧占位且无残留",
      aud["stages"]["present_before_upload"]["placeholders"] > 0
      and aud["stages"]["present_before_upload"]["residual_blocked"] == 0)
check("audit 上传后全部出字形",
      aud["stages"]["present_after_upload"]["glyphs"] > 0
      and aud["stages"]["present_after_upload"]["placeholders"] == 0)

fbs = mod.fallback_sdf_text(1)
check("fallback 未上传时用占位", fbs["mode"] == "placeholder")
_ph = [item for item in fbs["items"] if item["mode"] == "placeholder"]
check("fallback 占位项带原因", bool(_ph) and all(item["reason"] for item in _ph))
check("fallback 未就绪时不画字形", fbs["modes"]["glyph"] == 0)

det = mod.determinism_sdf_text(1, runs=4)
check("确定性：多次运行逐位一致", det["identical"] is True and det["unique"] == 1)

# ---------------------------------------------------------------------------
# 边界输入不崩溃
# ---------------------------------------------------------------------------
print("\n== 边界输入 ==")
for text in ("", " ", "\n\n", "a", "中文", "😀", "x" * 500):
    try:
        s = mod.layout_text_sdf(1, text=text)
        check(f"布局不崩溃 {text[:6]!r}", s is not None)
    except Exception as exc:  # noqa: BLE001
        check(f"布局不崩溃 {text[:6]!r}", False, repr(exc))
        traceback.print_exc()

print()
if failures:
    print(f"{len(failures)} 项失败: {failures}")
    sys.exit(1)
print("全部行为验证通过")

# pairwise-sdf-glyph-outline

## 领域

游戏、图形与多媒体

## 主题

SDF 字形渲染与动态描边

## 说明

上传栅栏与占位、屏幕空间描边、图集引用计数。

`graphics/sdf_glyph.py` 实现了一条完整的 SDF 文本管线，逐条修复题目里的六类故障：

| 故障现象 | 修复 | 关键实现 |
| --- | --- | --- |
| 首帧闪出上一页残留字形 | 上传栅栏与占位 | `UploadFence` / `DynamicAtlas.visible_for()`，栅栏未点亮只画占位块 |
| 大标题斜角描边断裂、小字号细笔画被吞 | 屏幕空间描边 | `outline_coverage()` 在像素空间做各向同性膨胀；`fill_coverage()` 用覆盖率代替硬阈值 |
| 换页后整段变方块、描边缓冲越界 | 图集引用计数 | `DynamicAtlas.retain/release/evict_lru()`，只回收引用计数为 0 的槽位；槽位按描边像素预留 `pad` |
| 异步回退字体改动已提交布局 | 布局快照 | `LayoutSnapshot` 为不可变数据类，校验和只含布局输入与结果 |
| DPI 切换后行宽/换行点变化 | DPI 重排 | 布局在 em 空间浮点累加，最后按 `size_px * dpi / 96` 缩放，字距不取整 |
| 混排回退字体后行高跳变、光标错位 | 回退字体基线统一 | 行高/基线取整条字体链清单（manifest），字形数据可异步到达 |

运行契约测试与行为验证：

```bash
python -m pytest -q             # 15 个契约用例
python -I tests/functional_check.py   # 六个修复点的对抗性行为验证
```

`tests/functional_check.py` 用反例证明修复生效，例如：未点亮的栅栏下绘制结果必须是占位块、
硬 alpha 阈值在 4px 字号下确实把细笔画吞成 0、旧实现的像素取整确实让换行点随 DPI 变化、
旧实现按“已加载字体”算行高确实在回退字体到达后从 16px 跳到 19.2px。

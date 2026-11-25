# pairwise-sdf-glyph-outline

## 领域

游戏、图形与多媒体

## 主题

SDF 字形渲染与动态描边

## 说明

上传栅栏与占位、屏幕空间描边、图集引用计数。

初始状态下 `graphics/sdf_glyph.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

# SWOT Raster 湖泊数据处理（已归档）

本目录包含从 SWOT 卫星数据提取湖泊 WSE/面积时序的处理代码。
该处理流程已完成，产出文件已迁移至 `data/swot/lake_timeseries.csv`。

## 原始命令（仅供记录）

```bash
# 完整处理
python -m src.process.main --prior data/prior/QB58_716_4.shp

# 测试模式
python -m src.process.main --prior data/prior/QB58_716_4.shp --test
```

## 目录说明

- `src/` — 处理代码（preprocess, QC, spatial_clip, timeseries 等）
- `data/raw/` — SWOT 原始数据（symlink → 实际位置，1.7T）
- `data/prior/` — 先验湖泊 shapefile
- `output/merged/` — 合并后的 TIF（symlink → 实际位置，41G）
- `output/lakes/` — 逐湖裁切结果（1~58 号目录）
- `output/analysis/` — 空间连续性分析
- `output/logs/` — 运行日志
- `skills/` — Claude skill 定义

## 注意事项

- `data/raw/` 和 `output/merged/` 为 symlink，数据仍在原始磁盘位置
- 如需重新运行，需要 `py10` conda 环境
- 处理约束和详细文档已随代码归档于 `archived/swot/doc/`

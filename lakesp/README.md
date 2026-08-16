# LakeSP 数据处理（已归档）

本项目已完成并归档。

## 归档日期

2026-03-17

## 描述

将 SWOT L2 HR LakeSP shapefile 数据按湖泊 ID 聚合，提取时序水位数据。

## 数据流

```
SWOT_Source/ (原始数据，8578个产品目录)
    ↓ step1_分湖泊聚合shp.py
SWOT_By_Lake/
    ├── Obs_Lakes/        (已识别湖泊，~418个湖泊目录)
    └── Unassigned_Lakes/ (未分配湖泊)
    ↓ 统计湖泊观测次数.py
湖泊统计汇总.csv
```

## 文件说明

- `scripts/` - 处理脚本
  - `step1_分湖泊聚合shp.py` - 主要聚合脚本
  - `统计湖泊观测次数.py` - 统计脚本
  - `download.sh` - 数据下载脚本
- `results/` - 处理结果和日志
- `SWOT_By_Lake/` - 按湖泊聚合的数据

## 核心算法

加权平均（水位计算）：基于不确定度的平方倒数加权

```python
weights = 1.0 / (uncertainties ** 2)
weighted_mean = sum(values * weights) / sum(weights)
```

## 数据过滤

- 质量过滤：保留 `quality_f ≤ 2`（未评估/好/可疑）
- 冰期过滤：保留 `ice_clim_f == 0 or -999`（无冰/缺失）

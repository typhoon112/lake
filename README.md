# SWOT lake processing code

本仓库保存湖泊研究中两套历史 SWOT 数据处理代码，便于追溯和复现。

## 目录

- `lakesp/`：处理 SWOT L2 HR LakeSP shapefile。代码按 `lake_id` 聚合多次过境记录，进行质量与冰期过滤，并生成逐湖面积和 WSE 时序。
- `raster/`：处理 SWOT L2 HR Raster NetCDF。代码包括预处理、质量控制、空间拼接与裁切、WSE 过滤、时序汇总及 area-WSE relationship 拟合。

## 运行入口

LakeSP 聚合：

```bash
python lakesp/scripts/step1_分湖泊聚合shp.py
```

运行前需要修改脚本顶部的 `INPUT_DIR` 和 `OUTPUT_DIR`。

Raster 完整流程：

```bash
cd raster
python -m src.process.main --prior data/prior/QB58_716_4.shp
```

详细设计、字段定义和运行说明见 `raster/docs/`。

## 数据说明

仓库仅包含代码和说明文档，不包含 SWOT 原始数据、先验湖泊矢量、处理结果、运行日志或 Python 缓存。原始路径配置保留了历史工作环境的写法，迁移到其他机器时需要相应调整。

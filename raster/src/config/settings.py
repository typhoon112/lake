"""
SWOT Raster 数据处理配置

基于用户确认的设计决策:
1. 空间处理: 筛选覆盖湖泊的granule文件，同一时刻的多个granule需要拼接
2. 面积分析: 需要，使用等面积投影
3. QC标准: SWOT官方标准 (cross_track > 10km)
4. 输出形式: TIF栅格 + SHP矢量双格式
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 路径配置
PATHS = {
    'prior': PROJECT_ROOT / 'data' / 'prior',
    'raw': PROJECT_ROOT / 'data' / 'raw',
    'output': PROJECT_ROOT / 'output',
    'logs': PROJECT_ROOT / 'logs',
}

# 填充值定义
FILL_VALUES = {
    'float32': 9.96921e+36,
    'uint8': 255,
    'uint32': 4294967295,
}

# 便捷访问常量
FILL_VALUE_FLOAT = FILL_VALUES['float32']
FILL_VALUE_UINT8 = FILL_VALUES['uint8']
FILL_VALUE_UINT32 = FILL_VALUES['uint32']

# 坐标系
CRS_WGS84 = "EPSG:4326"
CRS_ALBERS_ASIA = "EPSG:102025"  # Albers Asia 等面积投影
EQUAL_AREA_CRS = CRS_ALBERS_ASIA

# 统一处理投影（所有 granule 转换到此投影后处理）
# 使用 UTM Zone 46N，与先验湖泊边界一致，面积误差 < 0.3%
TARGET_CRS = "EPSG:32646"

# 质量过滤参数
QUALITY_CONFIG = {
    # 必须过滤（SWOT官方标准）
    'wse_qual_max': 2,              # good(0) 或 suspect(1)
    'water_area_qual_max': 2,       # 面积质量：good 或 suspect
    'cross_track_min': 10000,       # 排除近 nadir 伪影（用户确认）

    # 可配置过滤（默认不过滤，用户可根据需要调整）
    'water_frac_min': 0.0,          # 水体比例阈值（0表示不过滤）
    'min_valid_pixels': 10,        # 最小有效像元数（200像素 ≈ 2km² at 100m）
    'wse_uncert_max': 2.0,         # 最大不确定度 (m)

    # 可选过滤（D版本默认跳过）
    'ice_clim_flag_valid': [0, -999, 255],
    'ice_dyn_flag_valid': [0, -999, 255],
    'skip_ice_filter': True,        # D版本跳过冰期过滤
}

# 输出参数
OUTPUT_CONFIG = {
    # 输出格式
    'output_tif': True,              # 输出TIF栅格
    'output_shp': True,              # 输出SHP矢量
    'output_csv': True,              # 输出CSV汇总

    # TIF输出选项
    'tif_include_water_frac': True,  # TIF中包含water_frac通道
    'tif_nodata': -9999.0,           # TIF NoData值

    # 不确定度
    'include_uncertainty': True,
}

# WSE 像素级过滤参数
FILTER_CONFIG = {
    'mad_k': 1.0,                   # MAD 倍数（3.0 ≈ 正态分布 3-sigma）
    'max_half_range': 0.5,          # 半宽上限 (m)，确保总区间 ≤ 0.6m
    'min_pixels': 10,               # 最少有效像元数
}

# 面积-水位曲线拟合参数
CURVE_FIT_CONFIG = {
    'iqr_factor': 1.5,              # 箱线图异常值剔除倍数
    'min_cleaned_points': 3,        # 清洗后最少数据点
    'confidence_level': 0.95,       # 置信区间
    'plot_diagnostics': True,       # 输出诊断图
    'n_curve_samples': 200,         # 曲线单调性检验采样点数
}

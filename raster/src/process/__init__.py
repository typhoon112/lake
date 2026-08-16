"""
处理模块

包含数据探索、预处理、质量控制、空间裁切、时序提取等功能
"""

from .data_explorer import (
    SWOTRasterExplorer,
    explore_single_file,
)
from .preprocess import (
    SWOTRasterPreprocessor,
    preprocess_single_granule,
)
from .quality_control import (
    QualityController,
    QualityConfig,
    apply_quality_control,
)
from .spatial_clip import (
    SpatialClipper,
    load_prior_boundaries,
    clip_to_prior,
)
from .timeseries import (
    TimeseriesExtractor,
    TimeseriesAggregator,
)

__all__ = [
    # 数据探索
    'SWOTRasterExplorer',
    'explore_single_file',
    # 预处理
    'SWOTRasterPreprocessor',
    'preprocess_single_granule',
    # 质量控制
    'QualityController',
    'QualityConfig',
    'apply_quality_control',
    # 空间裁切
    'SpatialClipper',
    'load_prior_boundaries',
    'clip_to_prior',
    # 时序提取
    'TimeseriesExtractor',
    'TimeseriesAggregator',
]

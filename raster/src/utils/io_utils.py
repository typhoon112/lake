"""
IO工具模块

提供文件读写、NetCDF处理等IO工具函数
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import xarray as xr
import rioxarray


# 从统一配置导入填充值常量（单一来源）
from ..config.settings import FILL_VALUE_FLOAT, FILL_VALUE_UINT8, FILL_VALUE_UINT32


def is_valid_value(value: Union[float, np.ndarray],
                   fill_value: float = FILL_VALUE_FLOAT) -> Union[bool, np.ndarray]:
    """
    检查值是否为有效数据（非填充值、非NaN）

    Args:
        value: 要检查的值或数组
        fill_value: 填充值

    Returns:
        布尔值或布尔数组，True表示有效
    """
    if isinstance(value, np.ndarray):
        # 处理不同数据类型
        if np.issubdtype(value.dtype, np.floating):
            return (value != fill_value) & ~np.isnan(value) & (value < fill_value)
        elif np.issubdtype(value.dtype, np.integer):
            # 整数类型：检查是否为填充值
            if value.dtype == np.uint8:
                return value != FILL_VALUE_UINT8
            elif value.dtype == np.uint32:
                return value != FILL_VALUE_UINT32
            else:
                return value != 0  # 默认非零为有效
        elif value.dtype.kind in ['U', 'S', 'O']:
            # 字符串或对象类型：全部视为有效
            return np.ones(value.shape, dtype=bool)
        else:
            # 其他类型：尝试转换为浮点数
            try:
                float_val = value.astype(float)
                return (float_val != fill_value) & ~np.isnan(float_val) & (float_val < fill_value)
            except:
                return np.ones(value.shape, dtype=bool)
    else:
        if isinstance(value, (int, float)):
            return value != fill_value and value < fill_value and not np.isnan(value)
        else:
            return True  # 非数值类型默认有效


def parse_swot_filename(filename: str) -> Dict[str, str]:
    """
    解析SWOT L2_HR_Raster文件名

    文件名格式:
    SWOT_L2_HR_Raster_<resolution>_<grid>_<overlap>_x_x_x_<cycle>_<pass>_<scene>F_<start>_<end>_<crid>.nc

    示例:
    SWOT_L2_HR_Raster_100m_UTM01C_N_x_x_x_034_287_002F_20250618T233535_20250618T233556_PID0_01.nc

    Args:
        filename: 文件名（可以是完整路径）

    Returns:
        包含解析信息的字典
    """
    # 提取文件名（去除路径）
    basename = os.path.basename(filename)

    # 移除.nc后缀
    if basename.endswith('.nc'):
        basename = basename[:-3]

    parts = basename.split('_')

    result = {
        'filename': basename + '.nc',
        'full_path': filename if os.path.isabs(filename) else None,
        'product': '_'.join(parts[0:4]) if len(parts) > 3 else None,  # SWOT_L2_HR_Raster
        'resolution': parts[4] if len(parts) > 4 else None,  # 100m
        'grid': parts[5] if len(parts) > 5 else None,  # UTM01C
        'overlap': parts[6] if len(parts) > 6 else None,  # N
        'cycle': parts[10] if len(parts) > 10 else None,  # 034
        'pass': parts[11] if len(parts) > 11 else None,  # 287
        'scene': parts[12] if len(parts) > 12 else None,  # 002F
        'start_time': parts[13] if len(parts) > 13 else None,  # 20250618T233535
        'end_time': parts[14] if len(parts) > 14 else None,  # 20250618T233556
        'crid': parts[15] if len(parts) > 15 else None,  # PID0
        'version': parts[16] if len(parts) > 16 else None,  # 01
    }

    # 生成时间组ID（用于空间拼接分组）
    # 同一cycle、pass、crid的数据是同一时刻观测的
    result['group_id'] = f"{result['cycle']}_{result['pass']}_{result['crid']}"

    # 解析日期
    if result['start_time']:
        try:
            result['date'] = datetime.strptime(result['start_time'][:8], '%Y%m%d').date()
            result['datetime'] = datetime.strptime(result['start_time'], '%Y%m%dT%H%M%S')
        except ValueError:
            result['date'] = None
            result['datetime'] = None

    return result


def group_granules_by_time(granule_paths: List[str]) -> Dict[str, List[str]]:
    """
    将granule文件按时间组分组（同一时刻观测的文件分到一组）

    同一cycle、pass、crid的granule是同一时刻观测的，
    如果湖泊跨越多个granule，需要先拼接再处理

    Args:
        granule_paths: granule文件路径列表

    Returns:
        {group_id: [granule_path1, granule_path2, ...]}
    """
    groups = defaultdict(list)
    for path in granule_paths:
        info = parse_swot_filename(path)
        group_id = info.get('group_id')
        if group_id:
            groups[group_id].append(path)

    return dict(groups)


def scan_raster_files(data_dir: str,
                      pattern: str = "*.nc",
                      recursive: bool = True) -> List[str]:
    """
    扫描目录下的所有Raster文件

    Args:
        data_dir: 数据目录
        pattern: 文件匹配模式
        recursive: 是否递归搜索子目录

    Returns:
        文件路径列表
    """
    data_path = Path(data_dir)

    if recursive:
        files = list(data_path.rglob(pattern))
    else:
        files = list(data_path.glob(pattern))

    return [str(f) for f in files if f.is_file()]


def open_swot_raster(filepath: str,
                      variable: str = 'wse',
                      chunks: Optional[Dict] = None) -> xr.DataArray:
    """
    打开SWOT Raster NetCDF文件中的单个变量

    Args:
        filepath: NetCDF文件路径
        variable: 变量名
        chunks: 分块参数（用于大数据处理）

    Returns:
        xarray DataArray
    """
    # 使用rioxarray打开，保留空间信息
    da = rioxarray.open_rasterio(
        f"netcdf:{filepath}:{variable}",
        chunks=chunks,
        default_name=variable
    )

    return da


def open_swot_raster_dataset(filepath: str,
                              variables: Optional[List[str]] = None,
                              chunks: Optional[Dict] = None) -> xr.Dataset:
    """
    打开SWOT Raster NetCDF文件，返回多个变量

    Args:
        filepath: NetCDF文件路径
        variables: 变量名列表，如果为None则打开所有变量
        chunks: 分块参数

    Returns:
        xarray Dataset
    """
    if variables is None:
        # 默认打开的核心变量
        variables = [
            'wse', 'wse_qual', 'wse_uncert',
            'water_area', 'water_area_qual', 'water_frac',
            'cross_track', 'illumination_time',
            'ice_clim_flag', 'ice_dyn_flag',
            'latitude', 'longitude'
        ]

    datasets = []
    for var in variables:
        try:
            da = open_swot_raster(filepath, var, chunks)
            datasets.append(da)
        except Exception as e:
            print(f"Warning: Could not open variable {var}: {e}")

    if not datasets:
        raise ValueError(f"Could not open any variables from {filepath}")

    # 合并为Dataset
    ds = xr.merge(datasets, compat='override')

    return ds


def get_granule_bounds(filepath: str) -> Optional[Tuple[float, float, float, float]]:
    """
    获取granule文件的经纬度边界

    Args:
        filepath: NetCDF文件路径

    Returns:
        (minx, miny, maxx, maxy) WGS84边界，或None（如果无法获取）
    """
    try:
        # 读取经纬度变量
        with xr.open_dataset(filepath) as ds:
            if 'longitude' in ds and 'latitude' in ds:
                lon = ds['longitude'].values
                lat = ds['latitude'].values

                # 过滤填充值
                valid_lon = lon[is_valid_value(lon)]
                valid_lat = lat[is_valid_value(lat)]

                if len(valid_lon) > 0 and len(valid_lat) > 0:
                    return (valid_lon.min(), valid_lat.min(),
                            valid_lon.max(), valid_lat.max())
    except Exception as e:
        print(f"Warning: Could not get bounds from {filepath}: {e}")

    return None


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建

    Args:
        directory: 目录路径

    Returns:
        Path对象
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

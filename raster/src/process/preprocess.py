"""
预处理模块

处理坐标转换、填充值处理、数据有效性掩码生成等
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray
from rasterio.crs import CRS
from rasterio.enums import Resampling

from ..config.settings import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_UINT8,
    FILL_VALUE_UINT32,
    EQUAL_AREA_CRS,
    CRS_WGS84,
    TARGET_CRS,
)
from ..utils.io_utils import (
    parse_swot_filename,
    group_granules_by_time,
    is_valid_value,
)
from ..utils.geo_utils import (
    transform_bounds_to_crs,
    bounds_overlap,
)


class SWOTRasterPreprocessor:
    """SWOT Raster 数据预处理器"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化预处理器

        Args:
            config: 配置参数
        """
        self.config = config or {}

    def read_granule(self, filepath: str,
                     variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        读取单个granule文件

        Args:
            filepath: NetCDF文件路径
            variables: 要读取的变量列表，None则读取核心变量

        Returns:
            xarray Dataset
        """
        if variables is None:
            variables = [
                'wse', 'wse_qual', 'wse_qual_bitwise', 'wse_uncert',
                'water_area', 'water_area_qual', 'water_frac',
                'cross_track', 'illumination_time',
                'ice_clim_flag', 'ice_dyn_flag',
                'layover_impact', 'n_wse_pix',
                'latitude', 'longitude',
                'x', 'y',  # 坐标变量
            ]

        # 使用xarray直接打开NetCDF
        ds = xr.open_dataset(filepath)

        # 保留CRS变量（用于空间处理）
        crs_vars = ['crs', 'spatial_ref']
        for cv in crs_vars:
            if cv in ds and cv not in variables:
                variables = variables + [cv]

        # 选择需要的变量（只选择存在的变量）
        available_vars = [v for v in variables if v in ds]
        ds_subset = ds[available_vars]

        # 将数据加载到内存后关闭原始文件句柄，防止文件描述符耗尽
        ds_subset = ds_subset.load()
        ds.close()

        return ds_subset

    def get_crs_info(self, ds: xr.Dataset) -> Optional[CRS]:
        """
        从数据集中获取CRS信息

        Args:
            ds: xarray Dataset

        Returns:
            CRS对象或None
        """
        # 方法1: 检查spatial_ref变量
        if 'spatial_ref' in ds:
            return CRS.from_wkt(ds['spatial_ref'].attrs.get('crs_wkt', ''))

        # 方法2: 检查全局属性
        if 'crs' in ds.attrs:
            return CRS.from_string(ds.attrs['crs'])

        # 方法3: 检查grid_mapping属性
        for var in ds.data_vars:
            if 'grid_mapping' in ds[var].attrs:
                grid_mapping = ds[var].attrs['grid_mapping']
                if grid_mapping in ds:
                    return CRS.from_wkt(ds[grid_mapping].attrs.get('crs_wkt', ''))

        # 方法4: 从x/y坐标推断UTM zone
        # 这需要更多信息，暂时返回None
        return None

    def create_valid_mask(self, ds: xr.Dataset,
                          variable: str = 'wse') -> np.ndarray:
        """
        创建有效数据掩码

        Args:
            ds: xarray Dataset
            variable: 用于创建掩码的主变量

        Returns:
            布尔类型的有效数据掩码数组
        """
        if variable not in ds:
            raise ValueError(f"Variable {variable} not found in dataset")

        data = ds[variable].values

        # 根据数据类型选择填充值
        if data.dtype.kind in ['f']:  # 浮点数
            fill_value = FILL_VALUE_FLOAT
        elif data.dtype.kind in ['u']:  # 无符号整数
            if data.dtype.itemsize == 1:
                fill_value = FILL_VALUE_UINT8
            else:
                fill_value = FILL_VALUE_UINT32
        elif data.dtype.kind in ['i']:  # 有符号整数
            fill_value = FILL_VALUE_UINT32
        else:
            # 非数值类型，全部视为有效
            return np.ones(data.shape, dtype=bool)

        # 创建有效掩码
        valid_mask = is_valid_value(data, fill_value)

        return valid_mask

    def fill_to_nan(self, ds: xr.Dataset,
                    variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        将填充值替换为NaN

        Args:
            ds: xarray Dataset
            variables: 要处理的变量列表

        Returns:
            处理后的Dataset
        """
        if variables is None:
            variables = ['wse', 'wse_uncert', 'water_area', 'water_frac',
                         'cross_track', 'layover_impact']

        ds_processed = ds.copy()

        for var in variables:
            if var in ds_processed:
                data = ds_processed[var].values

                # 根据数据类型选择填充值
                if data.dtype.kind == 'f':
                    fill_value = FILL_VALUE_FLOAT
                else:
                    continue  # 非浮点数不处理

                # 替换填充值为NaN
                data = data.astype(float)
                data[~is_valid_value(data, fill_value)] = np.nan
                ds_processed[var].values = data

        return ds_processed

    def filter_by_bounds(self, ds: xr.Dataset,
                         bounds: Tuple[float, float, float, float],
                         src_crs: str = CRS_WGS84) -> xr.Dataset:
        """
        根据边界框筛选数据

        Args:
            ds: xarray Dataset
            bounds: (minx, miny, maxx, maxy) WGS84边界
            src_crs: 边界的坐标系

        Returns:
            裁剪后的Dataset
        """
        # 获取数据的CRS
        data_crs = self.get_crs_info(ds)

        if data_crs is None:
            # 如果无法获取CRS，尝试使用经纬度坐标
            if 'longitude' in ds and 'latitude' in ds:
                lon = ds['longitude'].values
                lat = ds['latitude'].values

                # 创建掩码
                mask = ((lon >= bounds[0]) & (lon <= bounds[2]) &
                        (lat >= bounds[1]) & (lat <= bounds[3]))

                # 应用掩码
                ds_filtered = ds.where(mask)
                return ds_filtered
            else:
                raise ValueError("Cannot determine CRS and no lat/lon available")

        # 转换边界到数据坐标系
        data_bounds = transform_bounds_to_crs(bounds, src_crs, str(data_crs))

        # 使用sel方法进行裁剪
        # 假设坐标维度名为x和y
        if 'x' in ds.dims and 'y' in ds.dims:
            ds_filtered = ds.sel(
                x=slice(data_bounds[0], data_bounds[2]),
                y=slice(data_bounds[1], data_bounds[3])
            )
        else:
            # 使用其他方法
            ds_filtered = ds

        return ds_filtered

    def get_spatial_bounds(self, ds: xr.Dataset) -> Optional[Tuple[float, float, float, float]]:
        """
        获取数据集的空间边界（WGS84）

        Args:
            ds: xarray Dataset

        Returns:
            (minx, miny, maxx, maxy) 或 None
        """
        if 'longitude' in ds and 'latitude' in ds:
            lon = ds['longitude'].values
            lat = ds['latitude'].values

            # 过滤NaN
            valid_lon = lon[~np.isnan(lon)]
            valid_lat = lat[~np.isnan(lat)]

            if len(valid_lon) > 0 and len(valid_lat) > 0:
                return (valid_lon.min(), valid_lat.min(),
                        valid_lon.max(), valid_lat.max())

        return None

    def reproject_to_target_crs(self, ds: xr.Dataset,
                                 target_crs: Optional[str] = None,
                                 resolution: Optional[float] = None) -> xr.Dataset:
        """
        将数据集投影转换到目标 CRS

        统一所有 granule 到同一投影，解决不同 UTM Zone 无法直接拼接的问题。

        Args:
            ds: 输入 Dataset
            target_crs: 目标 CRS，默认使用配置中的 TARGET_CRS (EPSG:32646)
            resolution: 输出分辨率（米），默认保持原始分辨率

        Returns:
            投影转换后的 Dataset
        """
        import warnings

        if target_crs is None:
            target_crs = TARGET_CRS

        # 1. 获取原始 CRS
        src_crs = self._get_dataset_crs_string(ds)

        if src_crs is None:
            print("Warning: Cannot determine source CRS, skipping reprojection")
            return ds

        # 2. 检查是否需要转换
        if self._compare_crs(src_crs, target_crs):
            # CRS 相同，无需转换，但需要清理 CRS 变量避免 "Multiple grid mappings exist" 错误
            # 删除旧的 CRS 变量
            crs_vars_to_remove = ['crs', 'spatial_ref', 'grid_mapping']
            for cv in crs_vars_to_remove:
                if cv in ds:
                    ds = ds.drop_vars(cv)

            # 清除所有变量的 grid_mapping 属性
            for var in ds.data_vars:
                if 'grid_mapping' in ds[var].attrs:
                    del ds[var].attrs['grid_mapping']

            # 重新写入 CRS
            ds = ds.rio.write_crs(src_crs, inplace=False)
            return ds

        # 3. 准备数据集
        # 确保 rioxarray 能识别 CRS
        if not hasattr(ds, 'rio') or ds.rio.crs is None:
            ds = ds.rio.write_crs(src_crs, inplace=False)

        # 4. 确定要转换的变量（二维数值变量）
        vars_to_reproject = []
        for var in ds.data_vars:
            if var in ['crs', 'spatial_ref', 'latitude', 'longitude']:
                continue
            if ds[var].ndim == 2 and ds[var].dims == ('y', 'x'):
                if np.issubdtype(ds[var].dtype, np.number):
                    vars_to_reproject.append(var)

        if not vars_to_reproject:
            return ds

        # 5. 确定输出分辨率
        if resolution is None:
            try:
                res = ds.rio.resolution()
                resolution = abs(res[0])  # 取 X 方向分辨率
            except Exception as e:
                print(f"Warning: Failed to get resolution, using default 100m: {e}")
                resolution = 100.0  # 默认 100m

        # 6. 执行投影转换
        try:
            # 使用 rioxarray 的 reproject 方法
            with warnings.catch_warnings():


                # 创建空的输出 Dataset
                reprojected_vars = {}

                for var in vars_to_reproject:
                    try:
                        da = ds[var].rio.reproject(
                            target_crs,
                            resolution=resolution,
                            nodata=np.nan,
                            resampling=Resampling.nearest
                        )
                        reprojected_vars[var] = da
                    except Exception as e:
                        print(f"Warning: Failed to reproject variable {var}: {e}")
                        continue

                if not reprojected_vars:
                    return ds

                # 合并所有转换后的变量
                first_var = list(reprojected_vars.keys())[0]
                coords = {'x': reprojected_vars[first_var]['x'],
                          'y': reprojected_vars[first_var]['y']}

                reprojected_ds = xr.Dataset(reprojected_vars, coords=coords)

                # 添加 CRS 信息
                reprojected_ds = reprojected_ds.rio.write_crs(target_crs, inplace=False)

                # 【关键修复】重新计算 latitude/longitude 网格
                # 后续的空间重叠检查依赖这些变量
                try:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)

                    x_vals = reprojected_ds['x'].values
                    y_vals = reprojected_ds['y'].values
                    xx, yy = np.meshgrid(x_vals, y_vals)
                    lon, lat = transformer.transform(xx, yy)

                    reprojected_ds['longitude'] = (('y', 'x'), lon.astype(np.float32))
                    reprojected_ds['latitude'] = (('y', 'x'), lat.astype(np.float32))
                except Exception as e:
                    print(f"Warning: Failed to compute lat/lon grid: {e}")

                # 复制全局属性
                reprojected_ds.attrs.update(ds.attrs)
                reprojected_ds.attrs['reprojected_from'] = str(src_crs)
                reprojected_ds.attrs['reprojected_to'] = str(target_crs)

                return reprojected_ds

        except Exception as e:
            print(f"Warning: Reprojection failed: {e}")
            return ds

    def _get_dataset_crs_string(self, ds: xr.Dataset) -> Optional[str]:
        """获取数据集的 CRS 字符串"""
        # 方法1: 从 rioxarray 获取
        if hasattr(ds, 'rio') and ds.rio.crs:
            return str(ds.rio.crs)

        # 方法2: 从 crs 变量获取
        if 'crs' in ds:
            crs_var = ds['crs']
            if 'crs_wkt' in crs_var.attrs:
                return crs_var.attrs['crs_wkt']
            if 'spatial_ref' in crs_var.attrs:
                return crs_var.attrs['spatial_ref']

        # 方法3: 从 spatial_ref 变量获取
        if 'spatial_ref' in ds:
            if 'crs_wkt' in ds['spatial_ref'].attrs:
                return ds['spatial_ref'].attrs['crs_wkt']

        # 方法4: 从 grid_mapping 属性获取
        for var in ds.data_vars:
            if 'grid_mapping' in ds[var].attrs:
                gm_name = ds[var].attrs['grid_mapping']
                if gm_name in ds:
                    gm_var = ds[gm_name]
                    if 'crs_wkt' in gm_var.attrs:
                        return gm_var.attrs['crs_wkt']
                    if 'spatial_ref' in gm_var.attrs:
                        return gm_var.attrs['spatial_ref']

        return None

    def _compare_crs(self, crs1: str, crs2: str) -> bool:
        """比较两个 CRS 是否相同"""
        try:
            from rasterio.crs import CRS as RasterioCRS

            c1 = RasterioCRS.from_user_input(crs1)
            c2 = RasterioCRS.from_user_input(crs2)

            return c1 == c2
        except Exception:
            # Fallback: 字符串比较
            return str(crs1).upper().strip() == str(crs2).upper().strip()


def preprocess_single_granule(filepath: str,
                              output_dir: Optional[str] = None,
                              config: Optional[Dict] = None) -> xr.Dataset:
    """
    预处理单个granule文件

    Args:
        filepath: NetCDF文件路径
        output_dir: 输出目录（可选）
        config: 配置参数

    Returns:
        预处理后的Dataset
    """
    preprocessor = SWOTRasterPreprocessor(config)

    # 读取数据
    ds = preprocessor.read_granule(filepath)

    # 填充值转NaN
    ds = preprocessor.fill_to_nan(ds)

    # 创建有效掩码
    ds['valid_mask'] = (('y', 'x'), preprocessor.create_valid_mask(ds))

    # 添加元数据
    file_info = parse_swot_filename(filepath)
    ds.attrs['source_file'] = filepath
    ds.attrs['cycle'] = file_info.get('cycle', '')
    ds.attrs['pass'] = file_info.get('pass', '')
    ds.attrs['date'] = str(file_info.get('date', ''))

    return ds


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <netcdf_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    ds = preprocess_single_granule(filepath)

    print(f"预处理完成: {filepath}")
    print(f"变量: {list(ds.data_vars)}")
    print(f"维度: {dict(ds.dims)}")

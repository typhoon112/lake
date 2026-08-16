"""
空间裁切模块

使用先验湖泊边界裁切栅格数据，生成湖泊掩膜
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
from shapely.geometry import box, mapping, shape
from rasterio.features import geometry_mask

# 抑制 rioxarray/rasterio 警告
warnings.filterwarnings('ignore', message='.*crs.*')
warnings.filterwarnings('ignore', message='.*grid_mapping.*')
warnings.filterwarnings('ignore', category=UserWarning, module='rioxarray')
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')

from ..config.settings import CRS_WGS84
from ..utils.geo_utils import (
    gdf_to_bounds,
    bounds_overlap,
    transform_bounds_to_crs,
)


class SpatialClipper:
    """空间裁切器"""

    def __init__(self, prior_gdf: gpd.GeoDataFrame):
        """
        初始化裁切器

        Args:
            prior_gdf: 先验湖泊边界的GeoDataFrame
        """
        self.prior_gdf = prior_gdf
        self.prior_bounds_wgs84 = gdf_to_bounds(prior_gdf)

    def clip_to_lake(self,
                     ds: xr.Dataset,
                     lake_id: Optional[str] = None,
                     buffer_meters: float = 0,
                     id_column: Optional[str] = None) -> Tuple[xr.Dataset, Dict]:
        """
        将数据集裁切到湖泊边界

        Args:
            ds: 输入Dataset
            lake_id: 湖泊ID（如果prior_gdf有多个湖泊）
            buffer_meters: 缓冲距离（米）
            id_column: ID列名（自动检测如果为None）

        Returns:
            (裁切后的Dataset, 裁切统计)
        """
        stats = {
            'lake_id': lake_id,
            'original_shape': ds.dims,
            'clipped_shape': None,
            'valid_pixels': 0,
        }

        # 自动检测ID列名
        if id_column is None:
            if 'lake_id' in self.prior_gdf.columns:
                id_column = 'lake_id'
            elif 'id_unified' in self.prior_gdf.columns:
                id_column = 'id_unified'

        # 获取目标湖泊几何
        if lake_id is not None and id_column is not None:
            lake_geom = self.prior_gdf[self.prior_gdf[id_column] == lake_id].geometry
        else:
            lake_geom = self.prior_gdf.geometry

        if lake_geom.empty:
            raise ValueError(f"Lake geometry not found for lake_id={lake_id}")

        # 获取数据集的CRS
        ds_crs = self._get_dataset_crs(ds)
        if ds_crs is None:
            raise ValueError("Cannot determine dataset CRS")

        # 转换先验几何到数据集坐标系
        prior_crs = self.prior_gdf.crs or CRS_WGS84
        if prior_crs != ds_crs:
            lake_geom_proj = lake_geom.to_crs(ds_crs)
        else:
            lake_geom_proj = lake_geom

        # 应用缓冲（如果需要）
        if buffer_meters > 0:
            lake_geom_proj = lake_geom_proj.buffer(buffer_meters)

        # 方法1: 使用rioxarray的clip功能
        try:
            ds_clipped = self._clip_with_rioxarray(ds, lake_geom_proj)
        except Exception as e:
            # rioxarray 失败，使用手动方法
            ds_clipped = self._clip_manual(ds, lake_geom_proj)

        # 裁切后空数据检查
        if ds_clipped.sizes.get('x', 1) == 0 or ds_clipped.sizes.get('y', 1) == 0:
            raise ValueError(f"No data overlap for lake_id={lake_id}")

        # 更新统计
        stats['clipped_shape'] = ds_clipped.dims
        if 'valid_mask' in ds_clipped:
            stats['valid_pixels'] = int(ds_clipped['valid_mask'].sum().values)

        return ds_clipped, stats

    def _get_dataset_crs(self, ds: xr.Dataset) -> Optional[str]:
        """获取数据集的CRS"""
        # 方法1: 检查rioxarray的CRS
        if hasattr(ds, 'rio') and ds.rio.crs:
            return str(ds.rio.crs)

        # 方法2: 从grid_mapping变量获取（SWOT标准方式）
        for var in ds.data_vars:
            if 'grid_mapping' in ds[var].attrs:
                gm_name = ds[var].attrs['grid_mapping']
                if gm_name in ds:
                    gm_var = ds[gm_name]
                    # 优先使用spatial_ref
                    if 'spatial_ref' in gm_var.attrs:
                        return gm_var.attrs['spatial_ref']
                    if 'crs_wkt' in gm_var.attrs:
                        return gm_var.attrs['crs_wkt']

        # 方法3: 直接检查crs变量
        if 'crs' in ds:
            if 'spatial_ref' in ds['crs'].attrs:
                return ds['crs'].attrs['spatial_ref']
            if 'crs_wkt' in ds['crs'].attrs:
                return ds['crs'].attrs['crs_wkt']

        # 方法4: 检查spatial_ref变量
        if 'spatial_ref' in ds:
            return ds['spatial_ref'].attrs.get('crs_wkt', None)

        # 方法5: 从全局属性获取
        if 'crs' in ds.attrs:
            return ds.attrs['crs']

        return None

    def _clip_with_rioxarray(self, ds: xr.Dataset, geometry) -> xr.Dataset:
        """使用rioxarray进行裁切"""
        # 获取并设置CRS
        crs_str = self._get_dataset_crs(ds)
        if crs_str is None:
            raise ValueError("Cannot determine dataset CRS for rioxarray")

        # 删除旧的 CRS 变量，避免 "Multiple grid mappings exist" 错误
        crs_vars_to_remove = ['crs', 'spatial_ref', 'grid_mapping']
        for cv in crs_vars_to_remove:
            if cv in ds:
                ds = ds.drop_vars(cv)

        # 写入CRS到数据集
        ds = ds.rio.write_crs(crs_str, inplace=False)

        # 获取几何边界（geometry已经是投影后的坐标系）
        bounds = geometry.total_bounds

        # 裁切到边界（抑制警告）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ds_clipped = ds.rio.clip_box(
                    minx=bounds[0], miny=bounds[1],
                    maxx=bounds[2], maxy=bounds[3]
                )
            except Exception:
                # clip_box可能因坐标不匹配失败，尝试手动裁切
                if 'x' in ds.dims and 'y' in ds.dims:
                    y_vals = ds.coords['y'].values
                    if len(y_vals) > 1 and y_vals[0] > y_vals[-1]:
                        # y 递减（标准 UTM 方向，北到南），slice 需要大值在前
                        ds_clipped = ds.sel(
                            x=slice(bounds[0], bounds[2]),
                            y=slice(bounds[3], bounds[1])
                        )
                    else:
                        # y 递增
                        ds_clipped = ds.sel(
                            x=slice(bounds[0], bounds[2]),
                            y=slice(bounds[1], bounds[3])
                        )
                else:
                    ds_clipped = ds

        # 进一步用几何掩膜（抑制警告）
        shapes = [geom for geom in geometry.values]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ds_clipped = ds_clipped.rio.clip(shapes, geometry.crs)
            except Exception:
                # 手动创建掩膜
                ds_clipped = self._apply_geometry_mask(ds_clipped, geometry)

        return ds_clipped

    def _apply_geometry_mask(self, ds: xr.Dataset, geometry) -> xr.Dataset:
        """手动应用几何掩膜"""
        if 'x' not in ds.dims or 'y' not in ds.dims:
            return ds

        x_coords = ds['x'].values
        y_coords = ds['y'].values

        # 空数组保护：裁切后无有效数据时直接返回
        if len(x_coords) == 0 or len(y_coords) == 0:
            return ds

        # 创建Transform
        from rasterio.transform import from_bounds
        transform = from_bounds(
            x_coords.min(), y_coords.min(),
            x_coords.max(), y_coords.max(),
            len(x_coords), len(y_coords)
        )

        # 生成掩膜
        shapes = [mapping(geom) for geom in geometry.values]
        mask = geometry_mask(
            shapes,
            transform=transform,
            invert=True,
            out_shape=(len(y_coords), len(x_coords))
        )

        # 应用掩膜到数据
        ds_masked = ds.where(mask)

        return ds_masked

    def _clip_manual(self, ds: xr.Dataset, geometry) -> xr.Dataset:
        """手动裁切（备用方法）"""
        # 获取几何边界
        bounds = geometry.total_bounds  # (minx, miny, maxx, maxy)

        # 裁切到边界
        if 'x' in ds.dims and 'y' in ds.dims:
            y_vals = ds.coords['y'].values
            if len(y_vals) > 1 and y_vals[0] > y_vals[-1]:
                # y 递减（标准 UTM 方向）
                ds_clipped = ds.sel(
                    x=slice(bounds[0], bounds[2]),
                    y=slice(bounds[3], bounds[1])
                )
            else:
                ds_clipped = ds.sel(
                    x=slice(bounds[0], bounds[2]),
                    y=slice(bounds[1], bounds[3])
                )
        else:
            ds_clipped = ds

        # 应用几何掩膜
        ds_clipped = self._apply_geometry_mask(ds_clipped, geometry)

        return ds_clipped

    def check_overlap(self, ds: xr.Dataset) -> bool:
        """
        检查数据集是否与先验边界重叠

        Args:
            ds: 输入Dataset

        Returns:
            是否重叠
        """
        # 获取数据集边界
        if 'longitude' in ds and 'latitude' in ds:
            lon = ds['longitude'].values
            lat = ds['latitude'].values

            valid_lon = lon[~np.isnan(lon)]
            valid_lat = lat[~np.isnan(lat)]

            if len(valid_lon) > 0 and len(valid_lat) > 0:
                data_bounds = (valid_lon.min(), valid_lat.min(),
                               valid_lon.max(), valid_lat.max())
                return bounds_overlap(data_bounds, self.prior_bounds_wgs84)

        return False


def load_prior_boundaries(filepath: str) -> gpd.GeoDataFrame:
    """
    加载先验湖泊边界

    保持原始 CRS，clip_to_lake() 内部会自动转换到数据集 CRS。

    Args:
        filepath: Shapefile路径

    Returns:
        GeoDataFrame
    """
    gdf = gpd.read_file(filepath)

    # 确保有坐标系
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)

    return gdf


def clip_to_prior(ds: xr.Dataset,
                  prior_gdf: gpd.GeoDataFrame,
                  lake_id: Optional[str] = None) -> Tuple[xr.Dataset, Dict]:
    """
    便捷函数：裁切数据到先验边界

    Args:
        ds: 输入Dataset
        prior_gdf: 先验湖泊边界GeoDataFrame
        lake_id: 湖泊ID

    Returns:
        (裁切后的Dataset, 裁切统计)
    """
    clipper = SpatialClipper(prior_gdf)
    return clipper.clip_to_lake(ds, lake_id)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python spatial_clip.py <netcdf_file> <prior_shapefile>")
        sys.exit(1)

    nc_file = sys.argv[1]
    prior_file = sys.argv[2]

    # 加载先验边界
    prior_gdf = load_prior_boundaries(prior_file)
    print(f"加载先验边界: {len(prior_gdf)} 个要素")

    # 读取数据（这里需要先预处理）
    ds = xr.open_dataset(nc_file)
    print(f"加载数据: {nc_file}")

    # 裁切
    clipper = SpatialClipper(prior_gdf)
    if clipper.check_overlap(ds):
        ds_clipped, stats = clipper.clip_to_lake(ds)
        print(f"裁切完成: {stats}")
    else:
        print("数据与先验边界无重叠")

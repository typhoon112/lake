"""
WSE 像素级过滤模块

对裁切后的 TIF 数据进行像素级异常值过滤，提取可靠水位区间。
使用 MAD (Median Absolute Deviation) 方法，输出过滤后的 TIF/SHP。
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning, module='rioxarray')
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')

from ..config.settings import FILTER_CONFIG, OUTPUT_CONFIG


class WSEPixelFilter:
    """像素级 WSE 过滤器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or FILTER_CONFIG

    def compute_mad_bounds(self, values: np.ndarray,
                           k: Optional[float] = None) -> Tuple[float, float, float, float]:
        """
        基于 MAD 计算可靠区间，半宽不超过 max_half_range

        Args:
            values: 有效值数组
            k: MAD 倍数，默认使用配置值

        Returns:
            (median, lower, upper, scaled_mad)
        """
        k = k or self.config.get('mad_k', 3.0)
        max_hr = self.config.get('max_half_range', 0.3)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scaled_mad = mad * 1.482  # 缩放到与 std 等价

        if scaled_mad == 0:
            return median, median, median, 0.0

        half_range = min(k * scaled_mad, max_hr)
        lower = median - half_range
        upper = median + half_range
        return median, lower, upper, scaled_mad

    def filter_tif(self, tif_path: str) -> Optional[Dict]:
        """
        对单个 TIF 进行像素级过滤

        Args:
            tif_path: 输入 TIF 路径

        Returns:
            过滤结果字典，或 None（无有效数据）
        """
        ds = rioxarray.open_rasterio(tif_path)
        nodata = ds.rio.nodata if ds.rio.nodata is not None else -9999.0

        # 提取 WSE (band 1)
        wse = ds.sel(band=1).values.astype(float).copy()
        wse[wse == nodata] = np.nan
        valid_mask = ~np.isnan(wse) & (wse < 9e36)
        valid_wse = wse[valid_mask]

        if len(valid_wse) < self.config.get('min_pixels', 10):
            ds.close()
            return None

        # MAD 过滤
        median, lower, upper, scaled_mad = self.compute_mad_bounds(valid_wse)

        # 构建空间过滤掩膜
        filter_mask = (wse >= lower) & (wse <= upper) & valid_mask
        n_filtered = int(np.sum(filter_mask))

        if n_filtered == 0:
            ds.close()
            return None

        if n_filtered < self.config.get('min_pixels', 10):
            ds.close()
            return None

        # 过滤后中位数
        filtered_wse = float(np.median(wse[filter_mask]))

        # 准备过滤后数据
        wse_filtered = np.where(filter_mask, wse, np.nan)

        # 如果有 band2 (water_frac)，同步过滤
        has_water_frac = ds.shape[0] > 1
        if has_water_frac:
            wf = ds.sel(band=2).values.astype(float).copy()
            wf[wf == nodata] = np.nan
            wf_filtered = np.where(filter_mask, wf, np.nan)
            data = np.stack([wse_filtered, wf_filtered], axis=0)
            band_coords = [1, 2]
        else:
            data = wse_filtered[np.newaxis, :, :]
            band_coords = [1]

        # 构建输出 DataArray
        da = xr.DataArray(
            data,
            dims=['band', 'y', 'x'],
            coords={
                'band': band_coords,
                'y': ds.coords['y'],
                'x': ds.coords['x'],
            }
        )
        da.attrs['long_name'] = ['wse', 'water_frac'][:len(band_coords)]
        da = da.rio.write_crs(ds.rio.crs)
        da = da.rio.write_nodata(nodata, inplace=False)

        ds.close()

        return {
            'data': da,
            'wse_filtered': filtered_wse,
            'wse_lower': lower,
            'wse_upper': upper,
            'scaled_mad': scaled_mad,
            'valid_pixels_orig': int(np.sum(valid_mask)),
            'valid_pixels_filtered': n_filtered,
            'filter_retention': n_filtered / len(valid_wse) * 100,
        }

    def export_filtered_tif(self, da: xr.DataArray, output_path: str) -> str:
        """导出过滤后的 TIF"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        da.rio.to_raster(output_path)
        return output_path

    def export_filtered_shp(self, da: xr.DataArray, output_path: str,
                            lake_id: str, date: str, result: Dict) -> Optional[str]:
        """从过滤后栅格导出等高线 SHP"""
        from rasterio.features import shapes
        from shapely.geometry import shape as shp_shape
        from shapely.ops import unary_union
        import geopandas as gpd

        wse = da.sel(band=1).values
        mask = ~np.isnan(wse)
        if not mask.any():
            return None

        try:
            crs = da.rio.crs
            transform = da.rio.transform()
        except Exception:
            return None

        mask_uint8 = mask.astype(np.uint8)
        polygon_shapes = list(shapes(mask_uint8, mask=mask, transform=transform))
        geometries = [shp_shape(geom) for geom, value in polygon_shapes if value == 1]

        if not geometries:
            return None

        merged_geom = unary_union(geometries)
        gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=crs)
        gdf['lake_id'] = lake_id
        gdf['date'] = date
        gdf['wse'] = result['wse_filtered']
        gdf['wse_lower'] = result['wse_lower']
        gdf['wse_upper'] = result['wse_upper']
        gdf['valid_pix'] = result['valid_pixels_filtered']

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdf.to_file(output_path)
        return output_path


def run_wse_filter(output_dir: str,
                   config: Optional[Dict] = None,
                   lake_ids: Optional[List[str]] = None,
                   max_lakes: Optional[int] = None):
    """
    批量执行 WSE 过滤

    Args:
        output_dir: 输出根目录
        config: 过滤配置
        lake_ids: 指定湖泊 ID 列表（None 则处理全部）
        max_lakes: 最大处理湖泊数（测试用）
    """
    output_path = Path(output_dir)
    wse_filter = WSEPixelFilter(config)

    # 扫描所有湖泊的 rasters/ 目录
    raster_dirs = sorted([d for d in output_path.iterdir()
                          if d.is_dir() and (d / 'rasters').exists()])

    if not raster_dirs:
        print("未找到任何湖泊的 rasters/ 目录")
        return

    # 过滤指定湖泊
    if lake_ids:
        lake_id_strs = [str(lid) for lid in lake_ids]
        raster_dirs = [d for d in raster_dirs if d.name in lake_id_strs]

    if max_lakes:
        raster_dirs = raster_dirs[:max_lakes]

    print(f"发现 {len(raster_dirs)} 个湖泊待处理")

    # 读取原始 CSV
    csv_path = output_path / 'summary' / 'lake_timeseries.csv'
    if not csv_path.exists():
        print(f"未找到原始 CSV: {csv_path}")
        return

    df_orig = pd.read_csv(csv_path)
    records = []

    for lake_dir in tqdm(raster_dirs, desc="WSE 过滤"):
        lake_id = lake_dir.name
        raster_dir = lake_dir / 'rasters'
        tif_files = sorted(raster_dir.glob('*_wse.tif'))

        if not tif_files:
            continue

        filtered_raster_dir = lake_dir / 'rasters_filtered'
        filtered_vector_dir = lake_dir / 'vectors_filtered'

        for tif_path in tif_files:
            date = tif_path.stem.replace('_wse', '')
            result = wse_filter.filter_tif(str(tif_path))

            if result is None:
                continue

            # 导出过滤后 TIF
            out_tif = filtered_raster_dir / tif_path.name
            wse_filter.export_filtered_tif(result['data'], str(out_tif))

            # 导出过滤后 SHP
            out_shp = filtered_vector_dir / f"{date}_contour.shp"
            wse_filter.export_filtered_shp(
                result['data'], str(out_shp), lake_id, date, result
            )

            # 释放内存
            del result['data']

            # 查找原始 CSV 中对应的记录
            orig_row = df_orig[
                (df_orig['lake_id'].astype(str) == lake_id) & (df_orig['date'] == date)
            ]

            record = {
                'lake_id': int(lake_id) if lake_id.isdigit() else lake_id,
                'date': date,
                'wse_filtered': result['wse_filtered'],
                'wse_lower': result['wse_lower'],
                'wse_upper': result['wse_upper'],
                'scaled_mad': result['scaled_mad'],
                'valid_pixels_orig': result['valid_pixels_orig'],
                'valid_pixels_filtered': result['valid_pixels_filtered'],
                'filter_retention': round(result['filter_retention'], 1),
            }

            # 合并原始字段
            if not orig_row.empty:
                row = orig_row.iloc[0]
                for col in df_orig.columns:
                    if col not in record:
                        record[col] = row[col]

            records.append(record)

    # 导出过滤后 CSV
    if records:
        df_filtered = pd.DataFrame(records)

        # 调整列顺序：原始列在前，过滤字段在后
        orig_cols = [c for c in df_orig.columns if c in df_filtered.columns]
        filter_cols = [c for c in df_filtered.columns if c not in orig_cols]
        df_filtered = df_filtered[orig_cols + filter_cols]

        filtered_csv = output_path / 'summary' / 'lake_timeseries_filtered.csv'
        df_filtered.to_csv(filtered_csv, index=False)

        print(f"\n过滤完成!")
        print(f"  - 处理湖泊: {len(raster_dirs)}")
        print(f"  - 处理记录: {len(records)}")
        print(f"  - 平均保留率: {df_filtered['filter_retention'].mean():.1f}%")
        print(f"  - CSV: {filtered_csv}")
    else:
        print("无有效过滤结果")

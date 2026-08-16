"""
时序提取与输出模块

计算WSE统计量、水体面积，生成TIF栅格和SHP矢量双格式输出
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
from rasterio.features import shapes
from shapely.geometry import shape as shp_shape

from ..config.settings import (
    OUTPUT_CONFIG,
    EQUAL_AREA_CRS,
    CRS_WGS84,
)
from ..utils.geo_utils import calculate_area_in_equal_area


class TimeseriesExtractor:
    """时序提取器"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化提取器

        Args:
            config: 输出配置
        """
        self.config = config or OUTPUT_CONFIG

    def calculate_wse_statistics(self, ds: xr.Dataset,
                                  mask: Optional[np.ndarray] = None) -> Dict:
        """
        计算WSE统计量

        wse 使用不确定度加权平均作为主要值（统计最优），
        同时计算中位数作为备用。

        Args:
            ds: 数据集
            mask: 有效掩膜（可选）

        Returns:
            统计量字典
        """
        wse = ds['wse'].values

        # 构建统一的有效值掩膜（wse 和 wse_uncert 使用同一掩膜）
        if mask is not None:
            valid_mask = mask & ~np.isnan(wse)
        else:
            valid_mask = ~np.isnan(wse)
        valid_mask = valid_mask & (wse < 9e36)
        # 过滤 nodata 填充值（SWOT 用 -9999，rioxarray write_nodata 不改变原始值）
        nodata_val = -9999.0
        try:
            if hasattr(ds, 'rio') and ds.rio.nodata is not None:
                nodata_val = float(ds.rio.nodata)
        except Exception:
            pass
        valid_mask = valid_mask & (wse != nodata_val)

        valid_wse = wse[valid_mask]

        if len(valid_wse) == 0:
            return {
                'wse': None,
                'wse_median': None,
                'wse_std': None,
                'wse_uncert': None,
                'valid_pixels': 0,
            }

        # 中位数
        wse_median = float(np.nanmedian(valid_wse))

        # 标准差
        wse_std = float(np.nanstd(valid_wse))

        # 不确定度加权平均（主要值）
        if 'wse_uncert' in ds:
            wse_uncert_arr = ds['wse_uncert'].values
            valid_uncert = wse_uncert_arr[valid_mask]

            # 有效不确定度：正值、非NaN、合理范围
            uncert_valid = (valid_uncert > 0) & (~np.isnan(valid_uncert)) & (valid_uncert < 1000)
            if uncert_valid.any():
                weights = 1.0 / (valid_uncert[uncert_valid] ** 2)
                wse_val = float(np.nansum(valid_wse[uncert_valid] * weights) / np.nansum(weights))
                wse_uncert_combined = float(np.sqrt(1.0 / np.nansum(weights)))
                # std 基于与加权平均相同的像元子集
                wse_std = float(np.nanstd(valid_wse[uncert_valid]))
            else:
                wse_val = wse_median
                wse_uncert_combined = float(np.nanmedian(np.abs(valid_wse - wse_median)) * 1.482)
        else:
            wse_val = wse_median
            wse_uncert_combined = float(np.nanmedian(np.abs(valid_wse - wse_median)) * 1.482)

        return {
            'wse': wse_val,  # 不确定度加权平均（主要值）
            'wse_median': wse_median,
            'wse_std': wse_std,
            'wse_uncert': wse_uncert_combined,
            'valid_pixels': int(len(valid_wse)),
        }

    def calculate_water_area(self, ds: xr.Dataset,
                             mask: Optional[np.ndarray] = None,
                             pixel_size_m: Optional[float] = None) -> float:
        """
        计算水体面积

        基于有效像元数 × 像元面积的方法。如果未指定像元大小，
        尝试从数据集的空间信息自动获取。

        注意：使用二值掩膜（像素完全在内/外），边界像素的亚像元覆盖未考虑。
        对于大湖泊此误差可忽略，小湖泊建议参考 SWOT 产品的 water_area 变量。

        Args:
            ds: 数据集
            mask: 有效掩膜
            pixel_size_m: 像元大小（米），None 则自动从数据集获取

        Returns:
            水体面积（平方米）
        """
        # 自动获取分辨率（支持非正方形像元）
        try:
            res = ds.rio.resolution()
            pixel_size_x = abs(res[0])
            pixel_size_y = abs(res[1])
        except Exception:
            try:
                x = ds.coords['x'].values
                y = ds.coords['y'].values
                pixel_size_x = abs(float(x[1] - x[0])) if len(x) > 1 else 100.0
                pixel_size_y = abs(float(y[0] - y[1])) if len(y) > 1 else 100.0
            except Exception:
                pixel_size_x = pixel_size_y = 100.0

        # 如果指定了统一像元大小，覆盖自动检测结果
        if pixel_size_m is not None:
            pixel_size_x = pixel_size_m
            pixel_size_y = pixel_size_m

        # 基于有效 wse 像元数和分辨率估算面积
        if mask is not None:
            valid_count = int(np.sum(mask))
        else:
            wse = ds['wse'].values
            valid_count = int(np.sum(~np.isnan(wse) & (wse < 9e36)))

        return float(valid_count * pixel_size_x * pixel_size_y)

    def extract_contour_geometry(self, ds: xr.Dataset,
                                  mask: Optional[np.ndarray] = None) -> gpd.GeoDataFrame:
        """
        从栅格数据提取等高线几何

        Args:
            ds: 数据集
            mask: 有效掩膜

        Returns:
            包含等高线几何的GeoDataFrame
        """
        # 创建二值掩膜
        if mask is None:
            mask = ~np.isnan(ds['wse'].values)

        # 获取数据集的CRS和Transform
        try:
            crs = ds.rio.crs
            transform = ds.rio.transform()
        except:
            crs = CRS_WGS84
            transform = None

        # 将掩膜转换为多边形
        mask_uint8 = mask.astype(np.uint8)
        polygon_shapes = list(shapes(mask_uint8, mask=mask, transform=transform))

        # 创建几何列表
        geometries = []
        for geom, value in polygon_shapes:
            if value == 1:  # 只保留水体区域
                geometries.append(shp_shape(geom))

        if not geometries:
            return gpd.GeoDataFrame(geometry=[], crs=crs)

        # 合并所有多边形
        from shapely.ops import unary_union
        merged_geom = unary_union(geometries)

        # 创建GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=crs)

        return gdf

    def export_tif(self, ds: xr.Dataset,
                   output_path: str,
                   include_water_frac: bool = True) -> str:
        """
        导出TIF栅格文件

        Args:
            ds: 数据集
            output_path: 输出路径
            include_water_frac: 是否包含water_frac通道

        Returns:
            输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 准备输出数据
        if include_water_frac and 'water_frac' in ds:
            # 双通道输出: [wse, water_frac]
            wse = ds['wse'].values
            water_frac = ds['water_frac'].values

            # 创建双通道数组
            data = np.stack([wse, water_frac], axis=0)

            # 创建DataArray
            da = xr.DataArray(
                data,
                dims=['band', 'y', 'x'],
                coords={
                    'band': [1, 2],
                    'y': ds.coords['y'],
                    'x': ds.coords['x'],
                }
            )

            # 添加波段描述
            da.attrs['long_name'] = ['wse', 'water_frac']
            da.attrs['units'] = ['m', '-']
        else:
            # 单通道输出
            da = ds['wse']

        # 确保有CRS信息
        if not hasattr(da, 'rio') or da.rio.crs is None:
            # 尝试从数据集获取CRS
            if hasattr(ds, 'rio') and ds.rio.crs:
                da = da.rio.write_crs(ds.rio.crs)
            else:
                da = da.rio.write_crs(CRS_WGS84)

        # 设置 nodata 值（确保 GIS 软件正确识别无效像元）
        da = da.rio.write_nodata(self.config.get('tif_nodata', -9999.0), inplace=False)

        # 导出
        da.rio.to_raster(output_path)

        return output_path

    def export_shp(self, ds: xr.Dataset,
                   output_path: str,
                   lake_id: str,
                   date: str,
                   stats: Dict) -> str:
        """
        导出SHP矢量文件

        Args:
            ds: 数据集
            output_path: 输出路径
            lake_id: 湖泊ID
            date: 观测日期
            stats: 统计量字典

        Returns:
            输出文件路径
        """
        # 提取等高线几何
        gdf = self.extract_contour_geometry(ds)

        if gdf.empty:
            print(f"Warning: No valid geometry for {lake_id} on {date}")
            return None

        # 添加属性
        gdf['lake_id'] = lake_id
        gdf['date'] = date
        gdf['wse'] = stats.get('wse', None)
        gdf['wse_med'] = stats.get('wse_median', None)
        gdf['wse_std'] = stats.get('wse_std', None)
        gdf['wse_uncert'] = stats.get('wse_uncert', None)
        gdf['water_area'] = stats.get('water_area', None)
        gdf['valid_pix'] = stats.get('valid_pixels', 0)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 导出
        gdf.to_file(output_path)

        return output_path


class TimeseriesAggregator:
    """时序聚合器 - 管理多个时序点的数据，同日多条观测合并为一条"""

    def __init__(self, output_dir: str):
        """
        初始化聚合器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.extractor = TimeseriesExtractor()
        self.records = []  # 用于CSV汇总（最终合并后的记录）
        self._pending = {}  # 缓存同日多条观测: (lake_id, date) → {observations: [...]}

    def process_single_observation(self,
                                    ds: xr.Dataset,
                                    lake_id: str,
                                    date: str,
                                    mask: Optional[np.ndarray] = None,
                                    granule_id: Optional[str] = None) -> Dict:
        """
        缓存单个时序观测

        同一湖泊同一天的多条观测缓存到 _pending 中，
        在 _finalize_records() 阶段做像素级合并后统一输出。

        Args:
            ds: 数据集（已裁切到湖泊范围）
            lake_id: 湖泊ID
            date: 观测日期
            mask: 有效掩膜
            granule_id: 数据文件标识

        Returns:
            处理结果字典（含统计量，用于进度条显示）
        """
        # 计算统计量（用于进度条，最终值从合并数据重新计算）
        stats = self.extractor.calculate_wse_statistics(ds, mask)
        stats['water_area'] = self.extractor.calculate_water_area(ds, mask)

        # 缓存 ds 本身（延迟到 finalize 阶段合并后输出）
        dedup_key = (str(lake_id), str(date))
        if dedup_key not in self._pending:
            self._pending[dedup_key] = {
                'lake_id': str(lake_id),
                'date': date,
                'observations': [],
            }
        self._pending[dedup_key]['observations'].append({
            'ds': ds,
            'stats': stats,
            'granule_id': granule_id,
        })

        return {'stats': stats}

    def _merge_lake_datasets(self, ds_list: List[xr.Dataset]) -> xr.Dataset:
        """
        像素级合并同一湖泊同一天的多个裁切数据集

        重叠像素：wse 用不确定度加权平均，water_frac 用算术平均
        非重叠像素：保留单 pass 值

        Args:
            ds_list: 同日多个 pass 的裁切数据集

        Returns:
            合并后的数据集
        """
        if len(ds_list) == 1:
            return ds_list[0]

        # 确定输出范围（union of all bounds）
        all_bounds = []
        for ds in ds_list:
            try:
                all_bounds.append(ds.rio.bounds())
            except Exception:
                pass
        if not all_bounds:
            return ds_list[0]

        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        # 获取分辨率
        try:
            res = ds_list[0].rio.resolution()
            res_x, res_y = abs(res[0]), abs(res[1])
        except Exception:
            res_x = res_y = 100.0

        width = int(np.ceil((maxx - minx) / res_x))
        height = int(np.ceil((maxy - miny) / res_y))

        # 累加器
        # wse + wse_uncert: 不确定度加权平均
        out_wse_sum = np.zeros((height, width), dtype=np.float64)
        out_weight_sum = np.zeros((height, width), dtype=np.float64)
        # water_frac: 算术平均
        out_frac_sum = np.zeros((height, width), dtype=np.float64)
        out_frac_count = np.zeros((height, width), dtype=np.int32)

        from affine import Affine
        transform = Affine(res_x, 0, minx, 0, -res_y, maxy)

        for ds in ds_list:
            wse = ds['wse'].values.copy()
            uncert = ds['wse_uncert'].values.copy() if 'wse_uncert' in ds else np.full_like(wse, np.nan)

            # 处理 Y 轴方向：若 Y 递增则翻转为递减（与输出坐标系一致）
            y_vals = ds.coords.get('y')
            if y_vals is not None and len(y_vals) > 1 and y_vals.values[0] < y_vals.values[-1]:
                wse = wse[::-1, :]
                uncert = uncert[::-1, :]

            # 定位到输出数组中的偏移
            ds_bounds = ds.rio.bounds()
            col_off = int(round((ds_bounds[0] - minx) / res_x))
            row_off = int(round((maxy - ds_bounds[3]) / res_y))

            h, w = wse.shape
            # 裁剪到输出范围
            si = max(0, -row_off)
            ei = min(h, height - row_off)
            sj = max(0, -col_off)
            ej = min(w, width - col_off)
            if ei <= si or ej <= sj:
                continue

            src_wse = wse[si:ei, sj:ej]
            src_uncert = uncert[si:ei, sj:ej]

            valid = ~np.isnan(src_wse) & (src_wse < 9e36)

            slc = (slice(row_off + si, row_off + ei),
                   slice(col_off + sj, col_off + ej))

            # WSE 不确定度加权
            with np.errstate(divide='ignore', invalid='ignore'):
                weight = np.where(
                    valid & ~np.isnan(src_uncert) & (src_uncert > 0),
                    1.0 / src_uncert ** 2, 0.0
                )
            out_wse_sum[slc] = np.where(
                valid, out_wse_sum[slc] + src_wse * weight, out_wse_sum[slc]
            )
            out_weight_sum[slc] += weight

            # water_frac 算术平均（同样使用翻转后的数据）
            if 'water_frac' in ds:
                frac_data = ds['water_frac'].values.copy()
                if y_vals is not None and len(y_vals) > 1 and y_vals.values[0] < y_vals.values[-1]:
                    frac_data = frac_data[::-1, :]
                src_frac = frac_data[si:ei, sj:ej]
                frac_valid = valid & ~np.isnan(src_frac)
                out_frac_sum[slc] = np.where(
                    frac_valid, out_frac_sum[slc] + src_frac, out_frac_sum[slc]
                )
                out_frac_count[slc] += frac_valid.astype(np.int32)

        # 合并结果
        with np.errstate(divide='ignore', invalid='ignore'):
            merged_wse = np.where(
                out_weight_sum > 0, out_wse_sum / out_weight_sum, np.nan
            )
            merged_uncert = np.where(
                out_weight_sum > 0, 1.0 / np.sqrt(out_weight_sum), np.nan
            )
            merged_frac = np.where(
                out_frac_count > 0, out_frac_sum / out_frac_count, np.nan
            )

        # 构建 Dataset
        y_coords = np.linspace(maxy, maxy - (height - 1) * res_y, height)
        x_coords = np.linspace(minx, minx + (width - 1) * res_x, width)

        merged_ds = xr.Dataset({
            'wse': (('y', 'x'), merged_wse.astype(np.float32)),
            'wse_uncert': (('y', 'x'), merged_uncert.astype(np.float32)),
            'water_frac': (('y', 'x'), merged_frac.astype(np.float32)),
        }, coords={'y': y_coords, 'x': x_coords})

        crs = ds_list[0].rio.crs
        if crs:
            merged_ds = merged_ds.rio.write_crs(crs)
        merged_ds = merged_ds.rio.write_transform(transform)

        return merged_ds

    def flush_date(self, date: str):
        """
        刷新指定日期的缓存记录（按时间维度释放内存）

        在一个日期的所有时间组处理完毕后调用，
        将该日期的同日观测做像素级合并后输出文件并写入 records。

        Args:
            date: 日期字符串，如 '2025-08-27'
        """
        self._finalize_records(date_filter=date)

    def flush_all(self):
        """刷新所有缓存记录（用于处理结束时的最终输出）"""
        self._finalize_records()

    def _finalize_records(self, date_filter: Optional[str] = None):
        """
        将缓存中的同日观测做像素级合并后输出文件并写入 records

        Args:
            date_filter: 若指定，仅处理该日期的记录；None 则处理全部
        """
        if not self._pending:
            return

        # 确定本次要处理的 key
        if date_filter is not None:
            keys_to_process = [k for k in self._pending if k[1] == date_filter]
        else:
            keys_to_process = list(self._pending.keys())

        if not keys_to_process:
            return

        for dedup_key in keys_to_process:
            (lake_id, date) = dedup_key
            pending = self._pending[dedup_key]
            obs_list = pending['observations']
            obs_list = pending['observations']

            # 像素级合并
            ds_list = [o['ds'] for o in obs_list]
            if len(ds_list) == 1:
                merged_ds = ds_list[0]
                merged_stats = obs_list[0]['stats']
            else:
                merged_ds = self._merge_lake_datasets(ds_list)
                # 从合并后的 ds 重新计算所有统计量
                merged_stats = self.extractor.calculate_wse_statistics(merged_ds)
                merged_stats['water_area'] = self.extractor.calculate_water_area(merged_ds)

            # 无有效像元则跳过文件输出（避免空 TIF 和缺失 SHP 的不一致）
            if merged_stats.get('valid_pixels', 0) == 0:
                print(f"跳过 {lake_id} {date}: 无有效像元")
                continue

            # 创建输出目录
            lake_dir = self.output_dir / str(lake_id)
            lake_dir.mkdir(parents=True, exist_ok=True)

            # 输出 TIF（一个文件，不带 granule_id 后缀）
            tif_dir = lake_dir / 'rasters'
            tif_dir.mkdir(exist_ok=True)
            tif_path = tif_dir / f"{date}_wse.tif"
            self.extractor.export_tif(merged_ds, str(tif_path))

            # 输出 SHP（一个文件）
            shp_dir = lake_dir / 'vectors'
            shp_dir.mkdir(exist_ok=True)
            shp_path = shp_dir / f"{date}_contour.shp"
            self.extractor.export_shp(
                merged_ds, str(shp_path), str(lake_id), date, merged_stats
            )

            # 写入 record
            self.records.append({
                'lake_id': lake_id,
                'date': date,
                'wse': merged_stats.get('wse'),
                'wse_med': merged_stats.get('wse_median'),
                'wse_std': merged_stats.get('wse_std'),
                'wse_uncert': merged_stats.get('wse_uncert'),
                'water_area': merged_stats.get('water_area'),
                'valid_pixels': merged_stats.get('valid_pixels'),
                'granule_id': ','.join(
                    str(o['granule_id']) for o in obs_list if o['granule_id']
                ),
            })

            # 释放缓存的 ds
            for o in obs_list:
                if hasattr(o['ds'], 'close'):
                    o['ds'].close()

            # 从 pending 中移除已处理的 key
            del self._pending[dedup_key]

    def export_summary_csv(self) -> str:
        """
        导出汇总CSV文件

        Returns:
            CSV文件路径
        """
        import pandas as pd

        # 合并同日观测后生成 records
        self._finalize_records()

        if not self.records:
            return None

        df = pd.DataFrame(self.records)

        # 确保summary目录存在
        summary_dir = self.output_dir / 'summary'
        summary_dir.mkdir(parents=True, exist_ok=True)

        csv_path = summary_dir / 'lake_timeseries.csv'
        df.to_csv(csv_path, index=False)

        return str(csv_path)

    def export_statistics_json(self) -> str:
        """
        导出统计JSON文件

        Returns:
            JSON文件路径
        """
        import json

        # 确保记录已合并
        self._finalize_records()

        stats = {
            'total_observations': len(self.records),
            'lakes': list(set(r['lake_id'] for r in self.records)),
            'date_range': {
                'start': min(r['date'] for r in self.records) if self.records else None,
                'end': max(r['date'] for r in self.records) if self.records else None,
            },
            'records': self.records,
        }

        summary_dir = self.output_dir / 'summary'
        summary_dir.mkdir(parents=True, exist_ok=True)

        json_path = summary_dir / 'statistics.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

        return str(json_path)


if __name__ == '__main__':
    print("Timeseries extraction module loaded successfully.")
    print("Use TimeseriesAggregator to process multiple observations.")

"""
主处理流程

整合所有模块，执行完整的SWOT Raster数据处理流程
"""

import os
import sys
import argparse
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import xarray as xr
import geopandas as gpd
from tqdm import tqdm

# 抑制常见警告
warnings.filterwarnings('ignore', message='.*crs.*')
warnings.filterwarnings('ignore', message='.*grid_mapping.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pyogrio')
warnings.filterwarnings('ignore', category=UserWarning, module='rioxarray')
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pyogrio')
warnings.filterwarnings('ignore', message='.*too larger number.*')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    PATHS,
    OUTPUT_CONFIG,
    QUALITY_CONFIG,
    PROJECT_ROOT,
    TARGET_CRS,
)
from src.utils.io_utils import (
    scan_raster_files,
    parse_swot_filename,
    group_granules_by_time,
    ensure_dir,
)
from src.utils.geo_utils import bounds_overlap, gdf_to_bounds
from src.process.preprocess import SWOTRasterPreprocessor
from src.process.quality_control import QualityController
from src.process.spatial_clip import SpatialClipper, load_prior_boundaries
from src.process.timeseries import TimeseriesAggregator
from src.process.wse_filter import run_wse_filter


class SWOTRasterProcessor:
    """SWOT Raster 主处理器"""

    def __init__(self,
                 prior_path: str,
                 raster_dir: str,
                 output_dir: str,
                 config: Optional[Dict] = None):
        """
        初始化处理器

        Args:
            prior_path: 先验湖泊边界文件路径
            raster_dir: Raster数据目录
            output_dir: 输出目录
            config: 配置参数
        """
        self.prior_path = prior_path
        self.raster_dir = raster_dir
        self.output_dir = output_dir
        self.config = config or {}
        # 合并 QUALITY_CONFIG 中的默认值
        for key, val in QUALITY_CONFIG.items():
            if key not in self.config:
                self.config[key] = val

        # 初始化日志
        self.logger = self._setup_logger()

        # 初始化各模块
        self.prior_gdf = None
        self.preprocessor = SWOTRasterPreprocessor(config)
        self.quality_controller = QualityController()
        self.aggregator = None

        # 处理统计
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'observations_generated': 0,
            'errors': [],
            'success_count': 0,
            'skip_count': 0,
        }

    def _setup_logger(self) -> logging.Logger:
        """配置日志记录器"""
        # 创建日志目录
        log_dir = Path(self.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件名包含时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'process_{timestamp}.log'

        # 配置 logger（清除旧 handler 防止重复调用时累积）
        logger = logging.getLogger('SWOTRasterProcessor')
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        # 文件处理器 - 记录所有级别
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # 控制台处理器 - 只显示 INFO 及以上
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # 记录日志文件位置
        logger.info(f"日志文件: {log_file}")

        return logger

    def initialize(self):
        """初始化处理环境"""
        print("=" * 60)
        print("SWOT Raster 数据处理")
        print("=" * 60)

        # 加载先验边界
        print(f"\n[1/4] 加载先验湖泊边界: {self.prior_path}")
        self.prior_gdf = load_prior_boundaries(self.prior_path)
        print(f"  - 加载了 {len(self.prior_gdf)} 个湖泊要素")

        # 获取先验边界范围
        prior_bounds = gdf_to_bounds(self.prior_gdf)
        print(f"  - 空间范围 (WGS84): {prior_bounds}")

        # 创建输出目录
        ensure_dir(self.output_dir)
        self.aggregator = TimeseriesAggregator(self.output_dir)
        print(f"  - 输出目录: {self.output_dir}")

        return self

    def scan_and_filter_files(self) -> Dict[str, List[str]]:
        """
        扫描并筛选覆盖研究区的Raster文件

        Returns:
            按时间组分的文件字典
        """
        print(f"\n[2/4] 扫描Raster文件: {self.raster_dir}")

        # 扫描所有文件
        all_files = scan_raster_files(self.raster_dir)
        self.stats['total_files'] = len(all_files)
        print(f"  - 发现 {len(all_files)} 个NC文件")

        if len(all_files) == 0:
            raise ValueError(f"未在 {self.raster_dir} 找到任何NC文件")

        # 按时间分组
        time_groups = group_granules_by_time(all_files)
        print(f"  - 分为 {len(time_groups)} 个时间组")

        return time_groups

    def check_spatial_overlap(self, ds: xr.Dataset, prior_gdf: gpd.GeoDataFrame) -> bool:
        """
        检查数据集是否与先验边界重叠

        注意：此方法可能在 fill_to_nan 之前调用，
        lon/lat 可能仍含填充值 (~9.97e36)，需同时过滤 NaN 和填充值。

        Args:
            ds: xarray Dataset
            prior_gdf: 先验边界GeoDataFrame

        Returns:
            是否重叠
        """
        if 'longitude' not in ds or 'latitude' not in ds:
            return False

        lon = ds['longitude'].values
        lat = ds['latitude'].values

        # 同时过滤 NaN 和填充值（fill_to_nan 可能尚未调用）
        valid_lon = lon[~np.isnan(lon) & (np.abs(lon) <= 180)]
        valid_lat = lat[~np.isnan(lat) & (np.abs(lat) <= 90)]

        if len(valid_lon) == 0 or len(valid_lat) == 0:
            return False

        # 数据边界
        data_bounds = (valid_lon.min(), valid_lat.min(),
                       valid_lon.max(), valid_lat.max())

        # 先验边界
        prior_bounds = gdf_to_bounds(prior_gdf)

        # 检查重叠
        return bounds_overlap(data_bounds, prior_bounds)

    def merge_time_group(self,
                         group_id: str,
                         granule_paths: List[str],
                         prior_gdf: gpd.GeoDataFrame,
                         export_merged: bool = True) -> Tuple[xr.Dataset, str, str]:
        """
        阶段一：QC + 空间拼接 + 输出中间影像

        流程：
        1. 收集 QC 前统计
        2. 逐个 granule 进行 QC
        3. 空间拼接（重叠区取平均）
        4. 收集拼接后统计
        5. 输出三阶段对比日志
        6. [可选] 输出中间影像

        Args:
            group_id: 时间组ID
            granule_paths: granule文件路径列表
            prior_gdf: 先验边界GeoDataFrame
            export_merged: 是否输出中间影像

        Returns:
            (ds_merged, date_str, merged_path)
        """
        # ========== 阶段 1+2: 收集原始统计 + 逐个 granule QC ==========
        qc_filtered_datasets = []
        post_qc_stats = []
        pre_qc_stats = []

        for granule_path in granule_paths:
            ds = None
            try:
                # 读取
                ds = self.preprocessor.read_granule(granule_path)

                # 检查空间重叠
                if not self.check_spatial_overlap(ds, prior_gdf):
                    ds.close()
                    ds = None
                    continue

                # 填充值转NaN
                ds = self.preprocessor.fill_to_nan(ds)

                # 收集 QC 前统计（与主处理合并，避免重复读取）
                file_info = parse_swot_filename(granule_path)
                scene_id = file_info.get('scene', 'unknown')
                wse_raw = ds['wse'].values
                raw_valid = ~np.isnan(wse_raw) & (wse_raw < 9e36)
                if np.sum(raw_valid) > 0:
                    pre_qc_stats.append({
                        'scene': scene_id,
                        'wse_mean': float(np.nanmean(wse_raw[raw_valid])),
                        'wse_std': float(np.nanstd(wse_raw[raw_valid])),
                        'valid_pixels': int(np.sum(raw_valid))
                    })

                # 投影转换到统一 CRS (EPSG:32646)
                ds = self.preprocessor.reproject_to_target_crs(ds, TARGET_CRS)

                # 质量控制（坏像元变为 NaN）
                ds_qc, qc_stats = self.quality_controller.apply_quality_filter(ds)

                # 关闭原始数据集，释放内存
                ds.close()
                ds = None

                # 收集 QC 后统计
                wse = ds_qc['wse'].values
                valid_mask = ~np.isnan(wse) & (wse < 9e36)
                valid_pixels = int(np.sum(valid_mask))

                post_qc_stats.append({
                    'scene': scene_id,
                    'wse_mean': qc_stats.get('final_wse_mean'),
                    'valid_pixels': valid_pixels,
                    'retention_rate': qc_stats.get('retention_rate', 100.0)
                })

                qc_filtered_datasets.append(ds_qc)

            except Exception as e:
                self.stats['errors'].append({
                    'granule': Path(granule_path).name,
                    'error': str(e)
                })
            finally:
                if ds is not None:
                    ds.close()

        if not qc_filtered_datasets:
            return None, None, None

        # ========== 阶段 3: 空间拼接 ==========
        if len(qc_filtered_datasets) == 1:
            ds_merged = qc_filtered_datasets[0]
        else:
            ds_merged = self._merge_qc_datasets(qc_filtered_datasets)

        # 释放个体 QC 数据集内存（合并后不再需要）
        for ds_qc in qc_filtered_datasets:
            if ds_qc is not ds_merged:
                if hasattr(ds_qc, 'close'):
                    ds_qc.close()

        # ========== 阶段 4: 收集拼接后统计 ==========
        post_merge_stats = self._collect_merged_stats(ds_merged)

        # ========== 阶段 5: 输出对比日志 ==========
        self._log_processing_stats(group_id, pre_qc_stats, post_qc_stats, post_merge_stats)

        # 获取日期
        file_info = parse_swot_filename(granule_paths[0])
        date_str = str(file_info.get('date', 'unknown'))

        # ========== 阶段 6: [新增] 输出中间影像 ==========
        merged_path = None
        if export_merged:
            merged_path = self._export_merged_raster(ds_merged, group_id, date_str)

        return ds_merged, date_str, merged_path

    def clip_lakes_from_merged(self,
                               ds_merged: xr.Dataset,
                               group_id: str,
                               date_str: str,
                               prior_gdf: gpd.GeoDataFrame) -> List[Dict]:
        """
        阶段二：湖泊裁切 + WSE 计算

        Args:
            ds_merged: 拼接后的数据集
            group_id: 时间组ID
            date_str: 日期字符串
            prior_gdf: 先验边界GeoDataFrame

        Returns:
            处理结果列表
        """
        results = []

        if ds_merged is None:
            return results

        clipper = SpatialClipper(prior_gdf)

        # 预计算数据集空间边界（用于快速排除无重叠湖泊）
        try:
            bounds = ds_merged.rio.bounds()
            ds_bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)
        except Exception:
            ds_bbox = None

        # 判断湖泊ID列名
        if 'lake_id' in prior_gdf.columns:
            id_col = 'lake_id'
        elif 'id_unified' in prior_gdf.columns:
            id_col = 'id_unified'
        else:
            id_col = None

        if id_col:
            lake_ids = prior_gdf[id_col].unique()
        else:
            lake_ids = [f"lake_{i}" for i in range(len(prior_gdf))]

        lake_count = 0
        for lake_id in lake_ids:
            try:
                # 空间重叠预检：快速排除明显无重叠的湖泊
                if ds_bbox is not None and id_col:
                    lake_row = prior_gdf[prior_gdf[id_col] == lake_id]
                    if not lake_row.empty:
                        lake_bounds = lake_row.geometry.to_crs(
                            ds_merged.rio.crs
                        ).total_bounds
                        lake_bbox = (lake_bounds[0], lake_bounds[1],
                                     lake_bounds[2], lake_bounds[3])
                        if not bounds_overlap(ds_bbox, lake_bbox):
                            self.stats['skip_count'] += 1
                            continue

                # 裁切到湖泊边界
                # 裁切到湖泊边界
                ds_clipped, clip_stats = clipper.clip_to_lake(
                    ds_merged,
                    lake_id=lake_id,
                    id_column=id_col
                )

                # 检查是否有有效数据
                wse = ds_clipped['wse'].values
                valid_pixels = int(np.sum(~np.isnan(wse)))
                if valid_pixels < self.config.get('min_valid_pixels', 200):
                    self.stats['skip_count'] += 1
                    continue

                # 提取时序数据并输出
                result = self.aggregator.process_single_observation(
                    ds_clipped,
                    lake_id=str(lake_id),
                    date=date_str,
                    mask=None,
                    granule_id=group_id
                )

                results.append(result)
                self.stats['success_count'] += 1
                lake_count += 1

            except Exception as e:
                self.stats['skip_count'] += 1
                self.stats['errors'].append({
                    'lake_id': str(lake_id),
                    'granule': group_id,
                    'error': str(e)
                })

        # 输出湖泊处理统计
        if lake_count > 0:
            print(f"  湖泊处理: {lake_count}个湖泊, 生成{lake_count}条记录")

        return results

    def process_time_group(self,
                           group_id: str,
                           granule_paths: List[str],
                           prior_gdf: gpd.GeoDataFrame) -> List[Dict]:
        """
        处理单个时间组 - 完整流程（阶段一 + 阶段二）

        这是原有的入口方法，内部调用 merge_time_group 和 clip_lakes_from_merged，
        保持向后兼容性。

        Args:
            group_id: 时间组ID
            granule_paths: granule文件路径列表
            prior_gdf: 先验边界GeoDataFrame

        Returns:
            处理结果列表
        """
        # 阶段一：QC + 拼接（不输出中间影像）
        ds_merged, date_str, _ = self.merge_time_group(
            group_id, granule_paths, prior_gdf, export_merged=False
        )

        if ds_merged is None:
            return []

        try:
            # 阶段二：湖泊裁切 + WSE 计算
            results = self.clip_lakes_from_merged(ds_merged, group_id, date_str, prior_gdf)
            return results
        finally:
            if ds_merged is not None and hasattr(ds_merged, 'close'):
                ds_merged.close()

    def _merge_qc_datasets(self, datasets: List[xr.Dataset]) -> xr.Dataset:
        """合并 QC 后的数据集（重叠区取平均）

        对所有二维数值变量（wse, wse_uncert, water_frac 等）都进行空间合并，
        重叠区取平均值，确保跨 granule 的湖泊数据完整。
        """
        # 验证每个数据集的 CRS 是否正确
        for i, ds in enumerate(datasets):
            ds_crs = ds.rio.crs if hasattr(ds, 'rio') else None
            if ds_crs is None:
                raise ValueError(f"Dataset {i} lost CRS information after QC. "
                               f"This indicates a bug in the processing pipeline.")

            from rasterio.crs import CRS as RasterioCRS
            try:
                target = RasterioCRS.from_string(TARGET_CRS)
                if ds_crs != target:
                    raise ValueError(f"Dataset {i} has incorrect CRS: {ds_crs}, expected {TARGET_CRS}. "
                                   f"All granules should have been reprojected to {TARGET_CRS} before QC.")
            except ValueError:
                raise
            except Exception:
                pass

        # 确定需要合并的二维数值变量
        crs_vars = ['crs', 'spatial_ref', 'grid_mapping']
        coord_vars = ['x', 'y', 'longitude', 'latitude',
                      'illumination_time', 'illumination_time_tai']
        merge_vars = []
        for var in datasets[0].data_vars:
            if var in crs_vars or var in coord_vars:
                continue
            if datasets[0][var].ndim == 2 and datasets[0][var].dims == ('y', 'x'):
                if np.issubdtype(datasets[0][var].dtype, np.number):
                    merge_vars.append(var)

        # 对不同变量使用不同的合并策略
        merged_vars = {}
        first_merged = None

        # wse + wse_uncert: 不确定度加权平均 + RSS（一次遍历）
        has_wse = 'wse' in merge_vars
        has_uncert = 'wse_uncert' in merge_vars

        if has_wse and has_uncert:
            wse_arrays = [ds['wse'] for ds in datasets]
            uncert_arrays = [ds['wse_uncert'] for ds in datasets]
            merged_wse, merged_uncert = self._merge_wse_weighted(wse_arrays, uncert_arrays)
            merged_vars['wse'] = merged_wse
            merged_vars['wse_uncert'] = merged_uncert
            first_merged = merged_wse
        elif has_wse:
            merged_vars['wse'] = self._merge_with_average([ds['wse'] for ds in datasets])
            first_merged = merged_vars['wse']

        # 其他变量: 算术平均
        other_vars = [v for v in merge_vars if v not in ('wse', 'wse_uncert')]
        for var in other_vars:
            arrays = [ds[var] for ds in datasets]
            merged_vars[var] = self._merge_with_average(arrays)
            if first_merged is None:
                first_merged = merged_vars[var]

        # 重建 Dataset
        merged = xr.Dataset(merged_vars)

        # 确保 CRS 信息正确
        try:
            crs_to_use = datasets[0].rio.crs if hasattr(datasets[0], 'rio') and datasets[0].rio.crs else None
            if crs_to_use:
                for cv in crs_vars:
                    if cv in merged:
                        merged = merged.drop_vars(cv)
                merged = merged.rio.write_crs(crs_to_use, inplace=False)
        except Exception:
            pass

        return merged

    def _merge_with_average(self, arrays: List[xr.DataArray]) -> xr.DataArray:
        """使用算术平均合并多个数组"""
        from rasterio.crs import CRS
        from affine import Affine

        if len(arrays) == 1:
            return arrays[0]

        # 收集所有数组的边界和分辨率
        all_bounds = []
        for arr in arrays:
            try:
                bounds = arr.rio.bounds()
                all_bounds.append(bounds)
            except Exception as e:
                print(f"Warning: Failed to get bounds for array: {e}")

        if not all_bounds:
            return arrays[0]

        # 计算合并后的边界
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        # 获取分辨率（使用第一个数组）
        try:
            res = arrays[0].rio.resolution()
            res_x = abs(res[0])
            res_y = abs(res[1])
        except Exception:
            res_x = 100.0
            res_y = 100.0

        # 计算输出形状
        width = int(np.ceil((maxx - minx) / res_x))
        height = int(np.ceil((maxy - miny) / res_y))

        # 创建输出数组
        out_sum = np.zeros((height, width), dtype=np.float64)
        out_count = np.zeros((height, width), dtype=np.int32)

        # 创建输出 transform
        transform = Affine(res_x, 0, minx, 0, -res_y, maxy)

        # 获取 CRS
        try:
            crs = arrays[0].rio.crs
        except Exception:
            crs = CRS.from_epsg(4326)

        for arr in arrays:
            try:
                arr_data = arr.values
                arr_bounds = arr.rio.bounds()

                src_height, src_width = arr_data.shape

                col_offset = int(round((arr_bounds[0] - minx) / res_x))
                row_offset = int(round((maxy - arr_bounds[3]) / res_y))

                # 处理 Y 轴方向（可能需要翻转）
                arr_y = arr.coords.get('y', arr.coords.get('latitude'))
                if arr_y is not None and len(arr_y) > 1:
                    if arr_y.values[0] < arr_y.values[-1]:
                        arr_data = arr_data[::-1, :]

                # 累加到输出数组
                i_start = max(0, -row_offset)
                i_end = min(src_height, height - row_offset)
                j_start = max(0, -col_offset)
                j_end = min(src_width, width - col_offset)

                if i_end > i_start and j_end > j_start:
                    src_data = arr_data[i_start:i_end, j_start:j_end]
                    valid_mask = ~np.isnan(src_data) & (src_data < 9e36)

                    out_i_start = row_offset + i_start
                    out_i_end = row_offset + i_end
                    out_j_start = col_offset + j_start
                    out_j_end = col_offset + j_end

                    out_sum[out_i_start:out_i_end, out_j_start:out_j_end] = np.where(
                        valid_mask,
                        out_sum[out_i_start:out_i_end, out_j_start:out_j_end] + src_data,
                        out_sum[out_i_start:out_i_end, out_j_start:out_j_end]
                    )
                    out_count[out_i_start:out_i_end, out_j_start:out_j_end] += valid_mask.astype(np.int32)

            except Exception as e:
                print(f"Warning: Failed to merge array: {e}")
                continue

        # 计算平均值
        with np.errstate(divide='ignore', invalid='ignore'):
            out_avg = np.where(out_count > 0, out_sum / out_count, np.nan)

        # 创建输出 DataArray
        y_coords = np.linspace(maxy, maxy - (height - 1) * res_y, height)
        x_coords = np.linspace(minx, minx + (width - 1) * res_x, width)

        merged = xr.DataArray(
            out_avg,
            dims=['y', 'x'],
            coords={'y': y_coords, 'x': x_coords}
        )
        merged = merged.rio.write_crs(crs)
        merged = merged.rio.write_transform(transform)

        return merged

    def _merge_wse_weighted(self,
                            wse_arrays: List[xr.DataArray],
                            uncert_arrays: List[xr.DataArray]) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        合并 wse 和 wse_uncert（一次遍历）

        wse 使用不确定度加权平均: wse = Σ(wse_i/σ²_i) / Σ(1/σ²_i)
        wse_uncert 使用 RSS: σ = 1 / √(Σ(1/σ²_i))

        Returns:
            (merged_wse, merged_uncert)
        """
        from rasterio.crs import CRS
        from affine import Affine

        if len(wse_arrays) == 1:
            return wse_arrays[0], uncert_arrays[0]

        # 收集边界
        all_bounds = []
        for arr in wse_arrays:
            try:
                all_bounds.append(arr.rio.bounds())
            except Exception as e:
                print(f"Warning: Failed to get bounds for wse array: {e}")
        if not all_bounds:
            return wse_arrays[0], uncert_arrays[0]

        # 合并后的边界和分辨率
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        try:
            res = wse_arrays[0].rio.resolution()
            res_x, res_y = abs(res[0]), abs(res[1])
        except Exception:
            res_x, res_y = 100.0, 100.0

        width = int(np.ceil((maxx - minx) / res_x))
        height = int(np.ceil((maxy - miny) / res_y))

        # 累加器: Σ(wse/σ²), Σ(1/σ²)
        out_weighted_sum = np.zeros((height, width), dtype=np.float64)
        out_weight_sum = np.zeros((height, width), dtype=np.float64)

        transform = Affine(res_x, 0, minx, 0, -res_y, maxy)

        try:
            crs = wse_arrays[0].rio.crs
        except Exception:
            crs = CRS.from_epsg(4326)

        for idx, (wse_arr, uncert_arr) in enumerate(zip(wse_arrays, uncert_arrays)):
            try:
                wse_data = wse_arr.values
                uncert_data = uncert_arr.values
                arr_bounds = wse_arr.rio.bounds()

                src_height, src_width = wse_data.shape

                col_offset = int(round((arr_bounds[0] - minx) / res_x))
                row_offset = int(round((maxy - arr_bounds[3]) / res_y))

                # Y 轴翻转
                arr_y = wse_arr.coords.get('y', wse_arr.coords.get('latitude'))
                need_flip = False
                if arr_y is not None and len(arr_y) > 1:
                    if arr_y.values[0] < arr_y.values[-1]:
                        need_flip = True
                if need_flip:
                    wse_data = wse_data[::-1, :]
                    uncert_data = uncert_data[::-1, :]

                # 累加
                i_start = max(0, -row_offset)
                i_end = min(src_height, height - row_offset)
                j_start = max(0, -col_offset)
                j_end = min(src_width, width - col_offset)

                if i_end > i_start and j_end > j_start:
                    src_wse = wse_data[i_start:i_end, j_start:j_end]
                    src_uncert = uncert_data[i_start:i_end, j_start:j_end]
                    valid_mask = (~np.isnan(src_wse) & (src_wse < 9e36) &
                                  ~np.isnan(src_uncert) & (src_uncert > 0))

                    slc = (slice(row_offset + i_start, row_offset + i_end),
                           slice(col_offset + j_start, col_offset + j_end))

                    with np.errstate(divide='ignore', invalid='ignore'):
                        weight = np.where(valid_mask, 1.0 / (src_uncert ** 2), 0.0)
                    out_weighted_sum[slc] = np.where(
                        valid_mask,
                        out_weighted_sum[slc] + src_wse * weight,
                        out_weighted_sum[slc]
                    )
                    out_weight_sum[slc] += weight

            except Exception as e:
                print(f"Warning: Failed to merge wse array: {e}")
                continue

        # wse = Σ(wse*weight) / Σ(weight)
        # σ = 1 / √(Σ(weight))
        with np.errstate(divide='ignore', invalid='ignore'):
            merged_wse = np.where(out_weight_sum > 0,
                                  out_weighted_sum / out_weight_sum, np.nan)
            merged_uncert = np.where(out_weight_sum > 0,
                                     1.0 / np.sqrt(out_weight_sum), np.nan)

        # 创建输出坐标
        y_coords = np.linspace(maxy, maxy - (height - 1) * res_y, height)
        x_coords = np.linspace(minx, minx + (width - 1) * res_x, width)

        wse_da = xr.DataArray(merged_wse, dims=['y', 'x'],
                              coords={'y': y_coords, 'x': x_coords})
        wse_da = wse_da.rio.write_crs(crs)
        wse_da = wse_da.rio.write_transform(transform)

        uncert_da = xr.DataArray(merged_uncert, dims=['y', 'x'],
                                 coords={'y': y_coords, 'x': x_coords})
        uncert_da = uncert_da.rio.write_crs(crs)
        uncert_da = uncert_da.rio.write_transform(transform)

        return wse_da, uncert_da

    def _collect_merged_stats(self, ds: xr.Dataset) -> Dict:
        """收集拼接后的统计"""
        wse = ds['wse'].values
        valid_mask = ~np.isnan(wse) & (wse < 9e36)
        valid_wse = wse[valid_mask]

        if len(valid_wse) > 0:
            return {
                'wse_mean': float(np.nanmean(valid_wse)),
                'wse_std': float(np.nanstd(valid_wse)),
                'valid_pixels': int(len(valid_wse))
            }
        else:
            return {
                'wse_mean': None,
                'wse_std': None,
                'valid_pixels': 0
            }

    # ========== 分布式计算辅助方法 ==========

    # 中间影像包含的关键波段（顺序固定，用于多波段 TIF 的 band 映射）
    MERGED_BAND_VARS = ['wse', 'water_frac', 'wse_uncert']

    def _export_merged_raster(self, ds: xr.Dataset, group_id: str, date_str: str) -> str:
        """
        导出拼接后的中间影像（多波段 TIF）

        波段顺序: 1=wse, 2=water_frac, 3=wse_uncert

        Args:
            ds: 拼接后的数据集
            group_id: 时间组ID
            date_str: 日期字符串

        Returns:
            输出文件路径
        """
        merged_dir = Path(self.output_dir) / 'merged'
        merged_dir.mkdir(parents=True, exist_ok=True)

        output_path = merged_dir / f"{group_id}.tif"

        # 收集存在的波段
        band_vars = [v for v in self.MERGED_BAND_VARS if v in ds]
        if not band_vars:
            return None

        # 构建多波段数组
        bands = []
        for var in band_vars:
            bands.append(ds[var].values.astype(np.float32))

        data = np.stack(bands, axis=0)

        # 创建多波段 DataArray
        da = xr.DataArray(
            data,
            dims=['band', 'y', 'x'],
            coords={
                'band': list(range(1, len(band_vars) + 1)),
                'y': ds.coords['y'],
                'x': ds.coords['x'],
            }
        )
        da.attrs['long_name'] = band_vars

        # 写入 CRS
        crs = None
        if hasattr(ds, 'rio') and ds.rio.crs:
            crs = ds.rio.crs
        if crs:
            da = da.rio.write_crs(crs)

        # 设置 nodata 值（确保 GIS 软件正确识别无效像元）
        da = da.rio.write_nodata(OUTPUT_CONFIG['tif_nodata'], inplace=False)
        da.rio.to_raster(str(output_path))

        # 保存元数据（含波段映射）
        meta_path = merged_dir / f"{group_id}.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'group_id': group_id,
                'date': date_str,
                'band_vars': band_vars,
            }, f)

        return str(output_path)

    def _load_merged_raster(self, tif_path: Path) -> xr.Dataset:
        """
        读取中间影像（多波段 TIF），根据元数据映射波段名

        Args:
            tif_path: TIFF 文件路径

        Returns:
            xarray Dataset（含 wse, water_frac, wse_uncert 等）
        """
        import rioxarray

        da = rioxarray.open_rasterio(str(tif_path))

        # 尝试从元数据 JSON 获取波段名映射
        meta_path = tif_path.with_suffix('.json')
        band_vars = self.MERGED_BAND_VARS  # 默认值
        if meta_path.exists():
            try:
                meta = self._load_merged_metadata(meta_path)
                band_vars = meta.get('band_vars', band_vars)
            except Exception:
                pass
        elif 'long_name' in da.attrs:
            # 从 TIF 自身的 long_name 属性获取
            ln = da.attrs['long_name']
            if isinstance(ln, (list, tuple)):
                band_vars = list(ln)

        # 按 band 拆分为独立变量
        ds_vars = {}
        for i, var_name in enumerate(band_vars):
            if i < da.shape[0]:
                # sel 后必须 drop band 坐标，否则不同变量的 band 值不同
                # 放入同一个 Dataset 时会触发 MergeError: conflicting values
                ds_vars[var_name] = da.sel(band=da.coords['band'].values[i]).drop_vars('band')

        ds = xr.Dataset(ds_vars)

        # 写入 CRS
        if hasattr(da, 'rio') and da.rio.crs:
            ds = ds.rio.write_crs(da.rio.crs)

        return ds

    def _load_merged_metadata(self, meta_path: Path) -> Dict:
        """
        读取元数据

        Args:
            meta_path: JSON 元数据文件路径

        Returns:
            元数据字典
        """
        with open(meta_path) as f:
            meta = json.load(f)
        return meta

    def _log_processing_stats(self, group_id: str,
                               pre_qc: List[Dict],
                               post_qc: List[Dict],
                               post_merge: Dict) -> None:
        """输出三阶段处理统计"""
        print(f"\n{'='*70}")
        print(f"[处理统计] {group_id}")
        print(f"{'='*70}")

        # 阶段1 - QC前
        print(f"\n  阶段1 - QC前 (共 {len(pre_qc)} 个 scene):")
        for s in pre_qc:
            if s['wse_mean'] is not None:
                print(f"    Scene {s['scene']}: mean={s['wse_mean']:.3f}m, pixels={s['valid_pixels']:,}")

        # 阶段2 - QC后
        print(f"\n  阶段2 - QC后:")
        for s in post_qc:
            if s['wse_mean'] is not None:
                print(f"    Scene {s['scene']}: mean={s['wse_mean']:.3f}m, "
                      f"pixels={s['valid_pixels']:,}, retention={s['retention_rate']:.1f}%")

        # 阶段3 - 拼接后
        print(f"\n  阶段3 - 拼接后:")
        if post_merge['wse_mean'] is not None:
            print(f"    mean={post_merge['wse_mean']:.3f}m, "
                  f"total_pixels={post_merge['valid_pixels']:,}")
        print(f"{'='*70}")

    def run(self, max_files: Optional[int] = None):
        """
        运行完整处理流程

        Args:
            max_files: 最大处理文件数（用于测试）
        """
        # 初始化
        self.initialize()

        # 扫描文件
        time_groups = self.scan_and_filter_files()

        # 限制处理数量（测试用）
        if max_files:
            time_groups = dict(list(time_groups.items())[:max_files])
            print(f"\n[测试模式] 限制处理 {len(time_groups)} 个时间组")

        # 预计算每个日期对应的时间组数量（用于按日期 flush）
        date_group_count = {}
        for group_id, paths in time_groups.items():
            file_info = parse_swot_filename(paths[0])
            date = str(file_info.get('date', ''))
            if date:
                date_group_count[date] = date_group_count.get(date, 0) + 1

        date_processed_count = {}  # date → 已处理的时间组数

        # 处理每个时间组
        print(f"\n[3/4] 处理时间组")

        # 使用增强进度条
        pbar = tqdm(
            time_groups.items(),
            desc="处理中",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for group_id, granule_paths in pbar:
            results = self.process_time_group(group_id, granule_paths, self.prior_gdf)
            self.stats['processed_files'] += len(granule_paths)
            self.stats['observations_generated'] += len(results)

            # 按日期维度 flush：当一个日期的所有时间组处理完毕后，释放内存
            file_info = parse_swot_filename(granule_paths[0])
            date = str(file_info.get('date', ''))
            if date:
                date_processed_count[date] = date_processed_count.get(date, 0) + 1
                if date_processed_count[date] >= date_group_count.get(date, 1):
                    self.aggregator.flush_date(date)

            # 更新进度条描述
            pbar.set_postfix({
                '观测': self.stats['observations_generated'],
                '成功': self.stats['success_count'],
                '跳过': self.stats['skip_count']
            })

        # 导出汇总文件（flush 剩余记录）
        print(f"\n[4/4] 导出汇总文件")
        csv_path = self.aggregator.export_summary_csv()
        json_path = self.aggregator.export_statistics_json()

        if csv_path:
            print(f"  - CSV汇总: {csv_path}")
        if json_path:
            print(f"  - 统计JSON: {json_path}")

        # 打印最终统计
        self._print_final_stats()

        return self.stats

    def _print_final_stats(self):
        """打印最终统计信息"""
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)

        # 基本统计
        print(f"\n📊 处理统计:")
        print(f"  ├─ 总文件数: {self.stats['total_files']:,}")
        print(f"  ├─ 已处理: {self.stats['processed_files']:,}")
        print(f"  └─ 时间组数: {self.stats.get('time_groups', 0):,}")

        # 观测统计
        print(f"\n📈 观测统计:")
        print(f"  ├─ 生成观测数: {self.stats['observations_generated']:,}")
        print(f"  ├─ 成功处理: {self.stats['success_count']:,}")
        print(f"  └─ 跳过记录: {self.stats['skip_count']:,}")

        # 成功率
        total_attempts = self.stats['success_count'] + self.stats['skip_count']
        if total_attempts > 0:
            success_rate = self.stats['success_count'] / total_attempts * 100
            print(f"\n✅ 成功率: {success_rate:.1f}%")

        # 错误信息
        if self.stats['errors']:
            print(f"\n⚠️ 错误数: {len(self.stats['errors'])}")
            print("  前5个错误:")
            for err in self.stats['errors'][:5]:
                if 'lake_id' in err:
                    print(f"    - 湖泊 {err['lake_id']}: {err['error'][:50]}...")
                else:
                    print(f"    - {err.get('granule', 'unknown')}: {str(err.get('error', ''))[:50]}...")
            if len(self.stats['errors']) > 5:
                print(f"    ... 还有 {len(self.stats['errors']) - 5} 个错误")

            # 【新增】将完整错误日志写入文件
            self._write_error_log()

        # 输出文件
        print(f"\n📁 输出文件:")
        print(f"  ├─ CSV汇总: {self.output_dir}/summary/lake_timeseries.csv")
        print(f"  └─ 统计JSON: {self.output_dir}/summary/statistics.json")

    def _write_error_log(self):
        """将完整错误日志写入文件"""
        log_dir = Path(self.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        error_log_file = log_dir / f'errors_{timestamp}.log'

        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"SWOT Raster 处理错误日志\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总错误数: {len(self.stats['errors'])}\n")
            f.write("=" * 70 + "\n\n")

            for i, err in enumerate(self.stats['errors'], 1):
                f.write(f"[错误 {i}]\n")
                if 'lake_id' in err:
                    f.write(f"  湖泊ID: {err['lake_id']}\n")
                    f.write(f"  Granule: {err.get('granule', 'N/A')}\n")
                    f.write(f"  错误: {err['error']}\n")
                elif 'granule' in err:
                    f.write(f"  Granule: {err['granule']}\n")
                    f.write(f"  错误: {err.get('error', 'Unknown error')}\n")
                elif 'group_id' in err:
                    f.write(f"  时间组: {err['group_id']}\n")
                    f.write(f"  错误: {err.get('error', 'Unknown error')}\n")
                else:
                    f.write(f"  详细信息: {err}\n")
                f.write("\n")

        print(f"\n📝 完整错误日志: {error_log_file}")
        self.logger.info(f"错误日志已保存到: {error_log_file}")

    def run_stage1(self, max_files: Optional[int] = None):
        """
        阶段一入口：仅执行 QC + 拼接，输出中间影像

        用于分布式计算第一阶段，生成中间影像到 output/merged/

        Args:
            max_files: 最大处理文件数（用于测试）

        Returns:
            处理统计
        """
        # 初始化
        self.initialize()

        # 扫描文件
        time_groups = self.scan_and_filter_files()

        # 限制处理数量（测试用）
        if max_files:
            time_groups = dict(list(time_groups.items())[:max_files])
            print(f"\n[测试模式] 限制处理 {len(time_groups)} 个时间组")

        print(f"\n[阶段一] 空间拼接，输出中间影像到 output/merged/")

        # 创建输出目录
        merged_dir = Path(self.output_dir) / 'merged'
        merged_dir.mkdir(parents=True, exist_ok=True)

        merged_count = 0
        for group_id, granule_paths in tqdm(time_groups.items(), desc="拼接中"):
            try:
                ds_merged, date_str, merged_path = self.merge_time_group(
                    group_id, granule_paths, self.prior_gdf, export_merged=True
                )

                if ds_merged is not None:
                    merged_count += 1
                    self.stats['processed_files'] += len(granule_paths)

                    # 关闭数据集，释放内存
                    if hasattr(ds_merged, 'close'):
                        ds_merged.close()
                    del ds_merged

            except Exception as e:
                self.stats['errors'].append({
                    'group_id': group_id,
                    'error': str(e)
                })

        print(f"\n阶段一完成！")
        print(f"  - 生成中间影像: {merged_count} 个")
        print(f"  - 输出目录: {merged_dir}")

        self._print_final_stats()
        return self.stats

    def run_stage2(self):
        """
        阶段二入口：读取中间影像，执行湖泊裁切 + WSE 计算

        用于分布式计算第二阶段，从 output/merged/ 读取中间影像

        Returns:
            处理统计
        """
        # 初始化（只加载先验边界）
        print("=" * 60)
        print("SWOT Raster 数据处理 - 阶段二：湖泊裁切")
        print("=" * 60)

        print(f"\n[1/3] 加载先验湖泊边界: {self.prior_path}")
        self.prior_gdf = load_prior_boundaries(self.prior_path)
        print(f"  - 加载了 {len(self.prior_gdf)} 个湖泊要素")

        # 检查中间影像目录
        merged_dir = Path(self.output_dir) / 'merged'
        if not merged_dir.exists():
            raise ValueError(f"中间影像目录不存在: {merged_dir}")

        merged_files = list(merged_dir.glob('*.tif'))
        if not merged_files:
            raise ValueError(f"未找到中间影像文件: {merged_dir}")

        print(f"\n[2/3] 发现 {len(merged_files)} 个中间影像")

        # 预计算每个日期对应的文件数量（用于按日期 flush）
        date_file_count = {}
        for tif_path in merged_files:
            meta_path = tif_path.with_suffix('.json')
            if meta_path.exists():
                try:
                    meta = self._load_merged_metadata(meta_path)
                    date = str(meta.get('date', ''))
                    if date:
                        date_file_count[date] = date_file_count.get(date, 0) + 1
                except Exception:
                    pass

        date_processed_count = {}

        # 初始化聚合器
        self.aggregator = TimeseriesAggregator(self.output_dir)

        print(f"\n[3/3] 湖泊裁切 + WSE 计算")

        for tif_path in tqdm(merged_files, desc="裁切中"):
            group_id = tif_path.stem
            ds_merged = None
            try:
                # 读取中间影像
                ds_merged = self._load_merged_raster(tif_path)

                # 读取元数据
                meta_path = tif_path.with_suffix('.json')
                if meta_path.exists():
                    meta = self._load_merged_metadata(meta_path)
                    date_str = meta.get('date', 'unknown')
                else:
                    date_str = 'unknown'

                # 湖泊裁切
                results = self.clip_lakes_from_merged(
                    ds_merged, group_id, date_str, self.prior_gdf
                )
                self.stats['observations_generated'] += len(results)

                # 按日期维度 flush
                if date_str and date_str != 'unknown':
                    date_processed_count[date_str] = date_processed_count.get(date_str, 0) + 1
                    if date_processed_count[date_str] >= date_file_count.get(date_str, 1):
                        self.aggregator.flush_date(date_str)

            except Exception as e:
                self.stats['errors'].append({
                    'group_id': group_id,
                    'error': str(e)
                })
            finally:
                if ds_merged is not None:
                    if hasattr(ds_merged, 'close'):
                        ds_merged.close()
                    del ds_merged

        # 导出汇总文件（flush 剩余记录）
        print(f"\n[4/4] 导出汇总文件")
        csv_path = self.aggregator.export_summary_csv()
        json_path = self.aggregator.export_statistics_json()

        if csv_path:
            print(f"  - CSV汇总: {csv_path}")
        if json_path:
            print(f"  - 统计JSON: {json_path}")

        self._print_final_stats()
        return self.stats


def run_filter_stage(output_dir: str, max_lakes: Optional[int] = None):
    """
    阶段三：WSE 像素级过滤

    从已有的 output/{lake_id}/rasters/ 读取裁切后 TIF，
    执行 MAD 过滤，输出过滤后的 TIF/SHP/CSV。
    """
    print("=" * 60)
    print("WSE 像素级过滤 (MAD)")
    print("=" * 60)

    run_wse_filter(output_dir, max_lakes=max_lakes)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SWOT Raster 湖泊时序数据提取'
    )

    parser.add_argument(
        '--prior', '-p',
        type=str,
        required=True,
        help='先验湖泊边界Shapefile路径'
    )

    parser.add_argument(
        '--raster', '-r',
        type=str,
        default=None,
        help='Raster数据目录（默认使用配置中的路径）'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出目录（默认使用配置中的路径）'
    )

    parser.add_argument(
        '--max-files', '-m',
        type=int,
        default=None,
        help='最大处理文件数（用于测试）'
    )

    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='测试模式（只处理少量文件）'
    )

    parser.add_argument(
        '--stage', '-s',
        type=str,
        default='all',
        choices=['all', 'merge', 'clip', 'filter'],
        help='处理阶段: all=全部(默认), merge=仅拼接, clip=仅裁切, filter=WSE过滤'
    )

    args = parser.parse_args()

    # 设置路径
    raster_dir = args.raster or os.path.join(PROJECT_ROOT, PATHS['raw'])
    output_dir = args.output or os.path.join(PROJECT_ROOT, PATHS['output'])

    # 测试模式
    max_files = args.max_files
    if args.test:
        max_files = 3

    # 创建处理器
    processor = SWOTRasterProcessor(
        prior_path=args.prior,
        raster_dir=raster_dir,
        output_dir=output_dir
    )

    # 根据阶段参数选择运行方法
    if args.stage == 'all':
        stats = processor.run(max_files=max_files)
    elif args.stage == 'merge':
        stats = processor.run_stage1(max_files=max_files)
    elif args.stage == 'clip':
        stats = processor.run_stage2()
    elif args.stage == 'filter':
        output_dir = args.output or os.path.join(PROJECT_ROOT, PATHS['output'])
        n_lakes = 3 if args.test else None
        run_filter_stage(output_dir, max_lakes=n_lakes)
        return

    return stats


if __name__ == '__main__':
    main()

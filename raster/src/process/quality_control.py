"""
质量控制模块

实现质量标志过滤、空间伪影过滤、水体比例过滤等质量控制功能
"""

import numpy as np
import xarray as xr
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class QualityConfig:
    """质量控制配置（参数来源：src/config/settings.py → QUALITY_CONFIG）"""
    wse_qual_max: int
    water_area_qual_max: int
    cross_track_min: float
    water_frac_min: float
    min_valid_pixels: int
    wse_uncert_max: float
    ice_clim_flag_valid: List[int] = None
    ice_dyn_flag_valid: List[int] = None
    skip_ice_filter: bool = True

    def __post_init__(self):
        if self.ice_clim_flag_valid is None:
            self.ice_clim_flag_valid = [0, -999, 255]
        if self.ice_dyn_flag_valid is None:
            self.ice_dyn_flag_valid = [0, -999, 255]


class QualityController:
    """SWOT Raster 质量控制器"""

    def __init__(self, config: Optional[QualityConfig] = None):
        """
        初始化质量控制器

        Args:
            config: 质量控制配置，None 时从 settings.py 读取
        """
        if config is None:
            from ..config.settings import QUALITY_CONFIG
            config = QualityConfig(**QUALITY_CONFIG)
        self.config = config

    def apply_quality_filter(self, ds: xr.Dataset) -> Tuple[xr.Dataset, Dict]:
        """
        应用完整的质量控制过滤

        Args:
            ds: 输入Dataset

        Returns:
            (过滤后的Dataset, 过滤统计信息)
        """
        stats = {
            'initial_pixels': 0,
            'after_wse_qual': 0,
            'after_cross_track': 0,
            'after_water_frac': 0,
            'after_ice_filter': 0,
            'final_pixels': 0,
            'final_wse_mean': None,
            'retention_rate': 0.0,
        }

        # 获取初始有效像元数
        if 'valid_mask' in ds:
            stats['initial_pixels'] = int(ds['valid_mask'].sum().values)
        else:
            # 基于WSE变量
            wse = ds['wse'].values
            valid = ~np.isnan(wse) & (wse < 9e36)
            stats['initial_pixels'] = int(np.sum(valid))

        # 创建质量掩码
        quality_mask = self._create_quality_mask(ds)

        # 应用过滤（保留 CRS 信息）
        original_crs = ds.rio.crs if hasattr(ds, 'rio') and ds.rio.crs else None

        ds_filtered = ds.copy()
        ds_filtered['quality_mask'] = quality_mask

        # 恢复 CRS 信息（ds.copy() 会丢失 rioxarray 的 CRS 属性）
        if original_crs is not None:
            try:
                ds_filtered = ds_filtered.rio.write_crs(original_crs, inplace=False)
            except Exception:
                pass  # 静默失败，后续会处理

        # 【关键修复】将 QC 掩码应用到实际数据（坏像元设为 NaN）
        vars_to_filter = ['wse', 'wse_uncert', 'water_area', 'water_frac']
        for var in vars_to_filter:
            if var in ds_filtered:
                ds_filtered[var] = ds_filtered[var].where(quality_mask)

        # 【重要】.where() 操作会丢失 grid_mapping 属性，需要重新写入 CRS
        if original_crs is not None:
            try:
                # 删除旧的 CRS 变量，避免冲突
                for cv in ['crs', 'spatial_ref']:
                    if cv in ds_filtered:
                        ds_filtered = ds_filtered.drop_vars(cv)
                ds_filtered = ds_filtered.rio.write_crs(original_crs, inplace=False)
            except Exception:
                pass

        # 更新统计
        stats['final_pixels'] = int(quality_mask.sum().values)

        # 计算各阶段过滤的像元数
        self._calculate_filter_stats(ds, quality_mask, stats)

        return ds_filtered, stats

    def _create_quality_mask(self, ds: xr.Dataset) -> xr.DataArray:
        """
        创建质量控制掩码

        Args:
            ds: 输入Dataset

        Returns:
            布尔类型的DataArray，True表示通过质量控制
        """
        # 初始化掩码为全True
        shape = ds['wse'].shape
        mask = np.ones(shape, dtype=bool)

        # 1. WSE质量过滤
        mask = self._filter_by_wse_qual(ds, mask)

        # 2. 水体面积质量过滤
        mask = self._filter_by_water_area_qual(ds, mask)

        # 3. cross_track过滤
        mask = self._filter_by_cross_track(ds, mask)

        # 4. 水体比例过滤
        mask = self._filter_by_water_frac(ds, mask)

        # 5. 冰期过滤（可选）
        if not self.config.skip_ice_filter:
            mask = self._filter_by_ice_flags(ds, mask)

        # 6. 不确定度过滤
        mask = self._filter_by_uncertainty(ds, mask)

        return xr.DataArray(mask, dims=ds['wse'].dims)

    def _filter_by_wse_qual(self, ds: xr.Dataset, mask: np.ndarray) -> np.ndarray:
        """WSE质量标志过滤"""
        if 'wse_qual' not in ds:
            return mask

        wse_qual = ds['wse_qual'].values
        # 过滤无效值
        valid_qual = (wse_qual <= self.config.wse_qual_max) & (wse_qual < 255)

        return mask & valid_qual

    def _filter_by_water_area_qual(self, ds: xr.Dataset, mask: np.ndarray) -> np.ndarray:
        """水体面积质量标志过滤"""
        if 'water_area_qual' not in ds:
            return mask

        water_area_qual = ds['water_area_qual'].values
        valid_qual = (water_area_qual <= self.config.water_area_qual_max) & (water_area_qual < 255)

        return mask & valid_qual

    def _filter_by_cross_track(self, ds: xr.Dataset, mask: np.ndarray) -> np.ndarray:
        """cross_track空间伪影过滤"""
        if 'cross_track' not in ds:
            return mask

        cross_track = ds['cross_track'].values
        # 排除近nadir伪影
        valid_track = np.abs(cross_track) >= self.config.cross_track_min
        # 也排除无效值
        valid_track = valid_track & (np.abs(cross_track) < 1e10)

        return mask & valid_track

    def _filter_by_water_frac(self, ds: xr.Dataset, mask: np.ndarray) -> np.ndarray:
        """水体比例过滤"""
        if self.config.water_frac_min <= 0:
            return mask

        if 'water_frac' not in ds:
            return mask

        water_frac = ds['water_frac'].values
        valid_frac = (water_frac >= self.config.water_frac_min) & (water_frac <= 1.0)

        return mask & valid_frac

    def _filter_by_ice_flags(self, ds: xr.Dataset, mask: np.ndarray) -> np.ndarray:
        """冰期过滤"""
        result_mask = mask.copy()

        # 气候态冰标志
        if 'ice_clim_flag' in ds:
            ice_clim = ds['ice_clim_flag'].values
            valid_ice_clim = np.isin(ice_clim, self.config.ice_clim_flag_valid)
            result_mask = result_mask & valid_ice_clim

        # 动态冰标志
        if 'ice_dyn_flag' in ds:
            ice_dyn = ds['ice_dyn_flag'].values
            valid_ice_dyn = np.isin(ice_dyn, self.config.ice_dyn_flag_valid)
            result_mask = result_mask & valid_ice_dyn

        return result_mask

    def _filter_by_uncertainty(self, ds: xr.Dataset, mask: np.ndarray) -> np.ndarray:
        """不确定度过滤"""
        if self.config.wse_uncert_max <= 0:
            return mask

        if 'wse_uncert' not in ds:
            return mask

        wse_uncert = ds['wse_uncert'].values
        valid_uncert = (wse_uncert <= self.config.wse_uncert_max) & (wse_uncert < 1e10)

        return mask & valid_uncert

    def _calculate_filter_stats(self, ds: xr.Dataset,
                                 final_mask: xr.DataArray,
                                 stats: Dict) -> None:
        """计算各过滤阶段的统计"""
        shape = ds['wse'].shape

        # WSE质量过滤
        if 'wse_qual' in ds:
            wse_qual = ds['wse_qual'].values
            qual_mask = (wse_qual <= self.config.wse_qual_max) & (wse_qual < 255)
            stats['after_wse_qual'] = int(np.sum(qual_mask))

        # water_area_qual过滤
        if 'water_area_qual' in ds:
            waq = ds['water_area_qual'].values
            waq_mask = (waq <= self.config.water_area_qual_max) & (waq < 255)
            stats['after_water_area_qual'] = int(np.sum(waq_mask))

        # cross_track过滤
        if 'cross_track' in ds:
            cross_track = ds['cross_track'].values
            track_mask = np.abs(cross_track) >= self.config.cross_track_min
            track_mask = track_mask & (np.abs(cross_track) < 1e10)
            stats['after_cross_track'] = int(np.sum(track_mask))

        # 最终
        stats['final_pixels'] = int(final_mask.sum().values)

        # 计算 final_wse_mean 和 retention_rate
        if stats['final_pixels'] > 0 and 'wse' in ds:
            wse = ds['wse'].values
            valid_wse = wse[final_mask.values]
            valid_wse = valid_wse[~np.isnan(valid_wse)]
            if len(valid_wse) > 0:
                stats['final_wse_mean'] = float(np.nanmean(valid_wse))

        stats['retention_rate'] = stats['final_pixels'] / max(1, stats['initial_pixels']) * 100

    def get_filter_summary(self, stats: Dict) -> str:
        """
        生成过滤统计摘要

        Args:
            stats: 过滤统计字典

        Returns:
            摘要字符串
        """
        lines = ["质量控制过滤统计:"]
        lines.append(f"  初始有效像元: {stats['initial_pixels']:,}")

        if stats.get('after_wse_qual', 0) > 0:
            filtered = stats['initial_pixels'] - stats['after_wse_qual']
            lines.append(f"  WSE质量过滤: -{filtered:,} (剩余 {stats['after_wse_qual']:,})")

        if stats.get('after_cross_track', 0) > 0:
            prev = stats.get('after_wse_qual', stats['initial_pixels'])
            filtered = prev - stats['after_cross_track']
            lines.append(f"  cross_track过滤: -{filtered:,} (剩余 {stats['after_cross_track']:,})")

        final_filtered = stats['initial_pixels'] - stats['final_pixels']
        retention_rate = stats['final_pixels'] / max(1, stats['initial_pixels']) * 100

        lines.append(f"  最终有效像元: {stats['final_pixels']:,}")
        lines.append(f"  总过滤: -{final_filtered:,} (保留率 {retention_rate:.1f}%)")

        return "\n".join(lines)


def apply_quality_control(ds: xr.Dataset,
                          config: Optional[QualityConfig] = None) -> Tuple[xr.Dataset, Dict]:
    """
    便捷函数：应用质量控制

    Args:
        ds: 输入Dataset
        config: 质量控制配置

    Returns:
        (过滤后的Dataset, 过滤统计)
    """
    controller = QualityController(config)
    return controller.apply_quality_filter(ds)


if __name__ == '__main__':
    # 测试代码
    from preprocess import preprocess_single_granule
    import sys

    if len(sys.argv) < 2:
        print("Usage: python quality_control.py <netcdf_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    # 预处理
    ds = preprocess_single_granule(filepath)
    print(f"预处理完成，变量: {list(ds.data_vars)}")

    # 质量控制
    ds_filtered, stats = apply_quality_control(ds)

    # 打印统计
    controller = QualityController()
    print(controller.get_filter_summary(stats))

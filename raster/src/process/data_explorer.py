"""
数据探索模块

读取并解析SWOT L2_HR_Raster NetCDF文件结构，
提取元数据和变量信息，生成数据报告
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import xarray as xr
import rioxarray

from ..utils.io_utils import (
    parse_swot_filename,
    is_valid_value,
    FILL_VALUE_FLOAT,
    FILL_VALUE_UINT8,
    FILL_VALUE_UINT32,
)


class SWOTRasterExplorer:
    """SWOT L2_HR_Raster 数据探索器"""

    # 关键变量列表（按重要性排序）
    KEY_VARIABLES = [
        # 核心水位变量
        ('wse', 'float32', 'm', '水面高程 (EGM2008)'),
        ('wse_qual', 'uint8', '-', '水位质量标志'),
        ('wse_qual_bitwise', 'uint32', '-', '水位质量位标志'),
        ('wse_uncert', 'float32', 'm', '水位不确定度'),

        # 水体面积变量
        ('water_area', 'float32', 'm²', '水体面积'),
        ('water_area_qual', 'uint8', '-', '面积质量标志'),
        ('water_frac', 'float32', '-', '水体比例'),

        # 雷达后向散射
        ('sig0', 'float32', '-', '归一化雷达截面'),
        ('sig0_qual', 'uint8', '-', '雷达截面质量'),

        # 空间和时间
        ('cross_track', 'float32', 'm', '跨轨距离'),
        ('illumination_time', 'float64', 's', '观测时间'),
        ('latitude', 'float64', '°', '纬度'),
        ('longitude', 'float64', '°', '经度'),

        # 冰标志
        ('ice_clim_flag', 'uint8', '-', '气候态冰标志'),
        ('ice_dyn_flag', 'uint8', '-', '动态冰标志'),

        # 其他质量信息
        ('layover_impact', 'float32', 'm', '叠影影响'),
        ('n_wse_pix', 'uint32', '-', 'WSE像元数'),
        ('n_water_area_pix', 'uint32', '-', '面积像元数'),
        ('dark_frac', 'float32', '-', '暗水比例'),
    ]

    def __init__(self, filepath: str):
        """
        初始化探索器

        Args:
            filepath: NetCDF文件路径
        """
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.file_info = parse_swot_filename(filepath)
        self._ds = None
        self._metadata = None

    def _open_dataset(self) -> xr.Dataset:
        """打开NetCDF数据集"""
        if self._ds is None:
            self._ds = xr.open_dataset(self.filepath)
        return self._ds

    def _close_dataset(self):
        """关闭数据集"""
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取文件元数据

        Returns:
            元数据字典
        """
        if self._metadata is not None:
            return self._metadata

        ds = self._open_dataset()

        metadata = {
            # 文件信息
            'filepath': self.filepath,
            'filename': self.filename,
            'file_size_mb': os.path.getsize(self.filepath) / (1024 * 1024),

            # 解析的文件名信息
            'parsed_info': self.file_info,

            # 全局属性
            'global_attrs': dict(ds.attrs),

            # 数据维度
            'dimensions': {k: v for k, v in ds.dims.items()},

            # 坐标变量
            'coordinates': list(ds.coords.keys()),

            # 数据变量列表
            'variables': list(ds.data_vars.keys()),

            # 变量数量
            'n_variables': len(ds.data_vars),
        }

        # 提取CRS信息
        if 'spatial_ref' in ds:
            metadata['crs'] = ds['spatial_ref'].attrs.get('crs_wkt', 'Unknown')
        elif 'crs' in ds.attrs:
            metadata['crs'] = ds.attrs['crs']
        else:
            metadata['crs'] = 'UTM (需要从数据推断)'

        self._metadata = metadata
        return metadata

    def get_variable_info(self, var_name: str) -> Dict[str, Any]:
        """
        获取单个变量的详细信息

        Args:
            var_name: 变量名

        Returns:
            变量信息字典
        """
        ds = self._open_dataset()

        if var_name not in ds:
            return {'error': f'Variable {var_name} not found'}

        var = ds[var_name]

        info = {
            'name': var_name,
            'dtype': str(var.dtype),
            'shape': var.shape,
            'dims': list(var.dims),
            'attrs': dict(var.attrs),
        }

        # 获取数据统计
        data = var.values

        # 检查数据类型，只对数值类型计算统计
        if data.dtype.kind in ['f', 'i', 'u']:  # 浮点、整数、无符号整数
            valid_mask = is_valid_value(data)

            if valid_mask.any():
                valid_data = data[valid_mask].astype(float)  # 转换为浮点数
                info['statistics'] = {
                    'valid_count': int(valid_mask.sum()),
                    'fill_count': int((~valid_mask).sum()),
                    'min': float(np.nanmin(valid_data)),
                    'max': float(np.nanmax(valid_data)),
                    'mean': float(np.nanmean(valid_data)),
                    'std': float(np.nanstd(valid_data)),
                }
            else:
                info['statistics'] = {
                    'valid_count': 0,
                    'fill_count': data.size,
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None,
                }
        else:
            # 非数值类型（如字符串、字节等）
            info['statistics'] = {
                'valid_count': data.size,
                'fill_count': 0,
                'min': None,
                'max': None,
                'mean': None,
                'std': None,
                'note': f'Non-numeric type: {data.dtype}'
            }

        # 检查是否为填充值
        info['fill_value'] = var.attrs.get('_FillValue', None)

        return info

    def get_all_variables_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有变量的信息

        Returns:
            {变量名: 变量信息} 字典
        """
        ds = self._open_dataset()
        result = {}

        for var_name in ds.data_vars:
            result[var_name] = self.get_variable_info(var_name)

        return result

    def get_spatial_bounds(self) -> Optional[Dict[str, float]]:
        """
        获取数据的空间边界（WGS84）

        Returns:
            边界字典 {'minx', 'miny', 'maxx', 'maxy'} 或 None
        """
        ds = self._open_dataset()

        if 'longitude' in ds and 'latitude' in ds:
            lon = ds['longitude'].values
            lat = ds['latitude'].values

            lon_valid = lon[is_valid_value(lon)]
            lat_valid = lat[is_valid_value(lat)]

            if len(lon_valid) > 0 and len(lat_valid) > 0:
                return {
                    'minx': float(lon_valid.min()),
                    'maxx': float(lon_valid.max()),
                    'miny': float(lat_valid.min()),
                    'maxy': float(lat_valid.max()),
                }

        return None

    def generate_report(self) -> str:
        """
        生成数据探索报告（Markdown格式）

        Returns:
            Markdown格式的报告字符串
        """
        metadata = self.get_metadata()
        bounds = self.get_spatial_bounds()
        parsed = metadata['parsed_info']

        report = []
        report.append(f"# SWOT L2_HR_Raster 数据探索报告\n")

        # 文件信息
        report.append("## 文件信息\n")
        report.append(f"- **文件名**: {metadata['filename']}")
        report.append(f"- **文件大小**: {metadata['file_size_mb']:.2f} MB")
        report.append(f"- **维度**: {metadata['dimensions']}")
        report.append(f"- **变量数**: {metadata['n_variables']}")
        report.append(f"- **CRS**: {metadata['crs']}\n")

        # 时间信息
        if parsed.get('date'):
            report.append("## 时间信息\n")
            report.append(f"- **日期**: {parsed['date']}")
            report.append(f"- **Cycle**: {parsed['cycle']}")
            report.append(f"- **Pass**: {parsed['pass']}")
            report.append(f"- **Scene**: {parsed['scene']}")
            report.append(f"- **CRID**: {parsed['crid']}\n")

        # 空间信息
        if bounds:
            report.append("## 空间范围 (WGS84)\n")
            report.append(f"- **经度**: {bounds['minx']:.4f} ~ {bounds['maxx']:.4f}")
            report.append(f"- **纬度**: {bounds['miny']:.4f} ~ {bounds['maxy']:.4f}\n")

        # 关键变量统计
        report.append("## 关键变量统计\n")
        report.append("| 变量名 | 类型 | 形状 | 有效值数 | 范围 | 均值 | 标准差 |")
        report.append("|--------|------|------|----------|------|------|--------|")

        all_vars = self.get_all_variables_info()
        for var_name, expected_type, unit, desc in self.KEY_VARIABLES:
            if var_name in all_vars:
                var_info = all_vars[var_name]
                stats = var_info.get('statistics', {})

                if stats.get('valid_count', 0) > 0:
                    range_str = f"{stats['min']:.2f} ~ {stats['max']:.2f}"
                    mean_str = f"{stats['mean']:.2f}"
                    std_str = f"{stats['std']:.2f}"
                else:
                    range_str = "N/A"
                    mean_str = "N/A"
                    std_str = "N/A"

                report.append(
                    f"| {var_name} | {var_info['dtype']} | "
                    f"{var_info['shape']} | {stats.get('valid_count', 0):,} | "
                    f"{range_str} | {mean_str} | {std_str} |"
                )

        # 数据质量评估
        report.append("\n## 数据质量初步评估\n")

        # 检查WSE数据
        if 'wse' in all_vars:
            wse_stats = all_vars['wse'].get('statistics', {})
            valid_ratio = wse_stats.get('valid_count', 0) / max(1, np.prod(all_vars['wse']['shape']))
            report.append(f"- **WSE有效数据比例**: {valid_ratio:.1%}")

            # 检查WSE质量标志分布
            if 'wse_qual' in all_vars:
                qual_stats = all_vars['wse_qual'].get('statistics', {})
                report.append(f"- **WSE质量标志范围**: {qual_stats.get('min', 'N/A')} ~ {qual_stats.get('max', 'N/A')}")

        report.append("\n---")
        report.append(f"\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        self._close_dataset()
        return "\n".join(report)

    def save_report(self, output_path: str) -> None:
        """
        保存报告到文件

        Args:
            output_path: 输出文件路径
        """
        report = self.generate_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已保存到: {output_path}")


def explore_single_file(filepath: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    探索单个SWOT Raster文件

    Args:
        filepath: NetCDF文件路径
        output_dir: 输出目录（可选）

    Returns:
        探索结果字典
    """
    explorer = SWOTRasterExplorer(filepath)
    result = {
        'metadata': explorer.get_metadata(),
        'bounds': explorer.get_spatial_bounds(),
        'variables': explorer.get_all_variables_info(),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(filepath).stem}_report.md")
        explorer.save_report(output_path)

    return result


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_explorer.py <netcdf_file> [output_dir]")
        sys.exit(1)

    filepath = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    result = explore_single_file(filepath, output_dir)
    print(f"探索完成: {filepath}")
    print(f"变量数: {result['metadata']['n_variables']}")
    if result['bounds']:
        print(f"空间范围: {result['bounds']}")

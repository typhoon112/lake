"""
投影验证脚本

检查 SWOT Raster 数据的投影一致性：
1. 扫描所有 granule 的 CRS
2. 检查同一时间组内的 CRS 是否一致
3. 分析 UTM zone 分布
4. 输出详细报告

用法:
    python -m src.utils.check_projection --dir <data_dir> [--output report.json]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import xarray as xr
from tqdm import tqdm

from src.utils.io_utils import scan_raster_files, parse_swot_filename, group_granules_by_time


def extract_utm_zone(crs_wkt: str) -> Optional[str]:
    """
    从 CRS WKT 字符串中提取 UTM zone

    Args:
        crs_wkt: CRS WKT 字符串

    Returns:
        UTM zone 字符串，如 "46S", "47N" 等
    """
    if not crs_wkt:
        return None

    # 常见格式:
    # - "UTM zone 46S"
    # - "WGS 84 / UTM zone 46S"
    # - EPSG:32746

    wkt_upper = crs_wkt.upper()

    # 方法1: 从 WKT 中提取
    import re

    # 匹配 "UTM ZONE XXN" 或 "UTM ZONE XXS"
    match = re.search(r'UTM\s*ZONE\s*(\d{1,2})([NS])', wkt_upper)
    if match:
        return f"{match.group(1)}{match.group(2)}"

    # 方法2: 从 EPSG 代码推断
    # EPSG:32601-32660 = UTM North 1-60
    # EPSG:32701-32760 = UTM South 1-60
    match = re.search(r'EPSG[:\s]*(\d+)', crs_wkt)
    if match:
        epsg = int(match.group(1))
        if 32601 <= epsg <= 32660:
            return f"{epsg - 32600}N"
        elif 32701 <= epsg <= 32760:
            return f"{epsg - 32700}S"

    return None


def get_granule_crs(filepath: str) -> Dict:
    """
    获取单个 granule 的 CRS 信息

    Args:
        filepath: NetCDF 文件路径

    Returns:
        CRS 信息字典
    """
    result = {
        'filepath': filepath,
        'filename': Path(filepath).name,
        'crs_wkt': None,
        'utm_zone': None,
        'epsg': None,
        'spatial_ref': None,
        'grid_mapping_name': None,
        'x_range': None,
        'y_range': None,
        'error': None
    }

    try:
        ds = xr.open_dataset(filepath)

        # 方法1: 从 crs 变量获取
        if 'crs' in ds:
            crs_var = ds['crs']
            if 'crs_wkt' in crs_var.attrs:
                result['crs_wkt'] = crs_var.attrs['crs_wkt']
            if 'spatial_ref' in crs_var.attrs:
                result['spatial_ref'] = crs_var.attrs['spatial_ref']

        # 方法2: 从 spatial_ref 变量获取
        if 'spatial_ref' in ds and result['crs_wkt'] is None:
            if 'crs_wkt' in ds['spatial_ref'].attrs:
                result['crs_wkt'] = ds['spatial_ref'].attrs['crs_wkt']

        # 方法3: 从变量属性获取 grid_mapping
        for var in ds.data_vars:
            if 'grid_mapping' in ds[var].attrs:
                gm_name = ds[var].attrs['grid_mapping']
                result['grid_mapping_name'] = gm_name
                if gm_name in ds and 'crs_wkt' in ds[gm_name].attrs:
                    result['crs_wkt'] = ds[gm_name].attrs['crs_wkt']
                    break

        # 获取坐标范围
        if 'x' in ds and 'y' in ds:
            x = ds['x'].values
            y = ds['y'].values
            result['x_range'] = [float(x.min()), float(x.max())]
            result['y_range'] = [float(y.min()), float(y.max())]

        # 提取 UTM zone
        result['utm_zone'] = extract_utm_zone(result['crs_wkt'] or result['spatial_ref'] or '')

        # 尝试获取 EPSG
        if result['crs_wkt']:
            import re
            match = re.search(r'EPSG[:\s]*(\d+)', result['crs_wkt'].upper())
            if match:
                result['epsg'] = int(match.group(1))

        ds.close()

    except Exception as e:
        result['error'] = str(e)

    return result


def check_time_group_crs_consistency(time_groups: Dict[str, List[str]],
                                      sample_size: Optional[int] = None) -> Dict:
    """
    检查同一时间组内的 CRS 一致性

    Args:
        time_groups: 时间组字典
        sample_size: 抽样检查的数量（None 表示全部检查）

    Returns:
        检查结果字典
    """
    results = {
        'total_groups': len(time_groups),
        'consistent_groups': 0,
        'inconsistent_groups': 0,
        'single_granule_groups': 0,
        'inconsistent_details': [],
        'utm_zone_distribution': defaultdict(int),
        'sample_details': []
    }

    groups_to_check = list(time_groups.items())
    if sample_size:
        groups_to_check = groups_to_check[:sample_size]

    for group_id, granule_paths in tqdm(groups_to_check, desc="检查时间组"):
        if len(granule_paths) == 1:
            results['single_granule_groups'] += 1
            continue

        # 获取所有 granule 的 CRS
        crs_info_list = []
        utm_zones = set()

        for path in granule_paths:
            crs_info = get_granule_crs(path)
            crs_info_list.append(crs_info)
            if crs_info['utm_zone']:
                utm_zones.add(crs_info['utm_zone'])
                results['utm_zone_distribution'][crs_info['utm_zone']] += 1

        # 检查一致性
        if len(utm_zones) <= 1:
            results['consistent_groups'] += 1
        else:
            results['inconsistent_groups'] += 1
            results['inconsistent_details'].append({
                'group_id': group_id,
                'utm_zones': list(utm_zones),
                'granule_count': len(granule_paths),
                'details': [
                    {
                        'filename': c['filename'],
                        'utm_zone': c['utm_zone'],
                        'epsg': c['epsg']
                    }
                    for c in crs_info_list[:5]  # 只保留前5个
                ]
            })

        # 保存样本详情
        if len(results['sample_details']) < 10:
            results['sample_details'].append({
                'group_id': group_id,
                'granule_count': len(granule_paths),
                'utm_zones': list(utm_zones),
                'is_consistent': len(utm_zones) <= 1
            })

    return results


def analyze_utm_zones(data_dir: str, max_files: Optional[int] = None) -> Dict:
    """
    分析数据目录中所有文件的 UTM zone 分布

    Args:
        data_dir: 数据目录
        max_files: 最大文件数

    Returns:
        分析结果
    """
    print(f"\n扫描数据目录: {data_dir}")

    # 扫描文件
    all_files = scan_raster_files(data_dir)
    if max_files:
        all_files = all_files[:max_files]

    print(f"发现 {len(all_files)} 个文件")

    results = {
        'total_files': len(all_files),
        'utm_zone_by_filename': defaultdict(int),
        'utm_zone_by_content': defaultdict(int),
        'filename_content_mismatch': [],
        'crs_extraction_errors': [],
        'detailed_results': []
    }

    for filepath in tqdm(all_files, desc="分析 CRS"):
        # 从文件名提取 UTM zone
        filename = Path(filepath).name
        # 文件名格式: SWOT_L2_HR_Raster_100m_UTM46S_N_x_x_x_...
        parts = filename.split('_')
        utm_from_filename = None
        if len(parts) > 5:
            utm_part = parts[5]  # 如 "UTM46S"
            if utm_part.startswith('UTM'):
                utm_from_filename = utm_part[3:]  # "46S"

        results['utm_zone_by_filename'][utm_from_filename] += 1

        # 从文件内容提取 CRS
        crs_info = get_granule_crs(filepath)
        utm_from_content = crs_info['utm_zone']

        if utm_from_content:
            results['utm_zone_by_content'][utm_from_content] += 1

        # 检查一致性
        if utm_from_filename and utm_from_content:
            if utm_from_filename != utm_from_content:
                results['filename_content_mismatch'].append({
                    'filename': filename,
                    'filename_zone': utm_from_filename,
                    'content_zone': utm_from_content
                })

        if crs_info['error']:
            results['crs_extraction_errors'].append({
                'filename': filename,
                'error': crs_info['error']
            })

        # 保存详细结果（前20个）
        if len(results['detailed_results']) < 20:
            results['detailed_results'].append({
                'filename': filename,
                'utm_from_filename': utm_from_filename,
                'utm_from_content': utm_from_content,
                'epsg': crs_info['epsg'],
                'x_range': crs_info['x_range'],
                'y_range': crs_info['y_range']
            })

    return results


def generate_report(utm_analysis: Dict, group_analysis: Dict) -> str:
    """生成文本报告"""
    lines = []
    lines.append("=" * 80)
    lines.append("SWOT Rater 投影验证报告")
    lines.append("=" * 80)

    # UTM Zone 分布
    lines.append("\n## 1. UTM Zone 分布（从文件名）")
    lines.append("-" * 40)
    for zone, count in sorted(utm_analysis['utm_zone_by_filename'].items()):
        lines.append(f"  {zone}: {count:,} 个文件")

    lines.append("\n## 2. UTM Zone 分布（从文件内容）")
    lines.append("-" * 40)
    for zone, count in sorted(utm_analysis['utm_zone_by_content'].items()):
        lines.append(f"  {zone}: {count:,} 个文件")

    # 不一致问题
    if utm_analysis['filename_content_mismatch']:
        lines.append("\n## ⚠️ 3. 文件名与内容 CRS 不一致")
        lines.append("-" * 40)
        for item in utm_analysis['filename_content_mismatch'][:10]:
            lines.append(f"  {item['filename']}: 文件名={item['filename_zone']}, 内容={item['content_zone']}")
        if len(utm_analysis['filename_content_mismatch']) > 10:
            lines.append(f"  ... 还有 {len(utm_analysis['filename_content_mismatch']) - 10} 个")

    # 时间组一致性
    lines.append("\n## 4. 时间组 CRS 一致性检查")
    lines.append("-" * 40)
    lines.append(f"  总时间组数: {group_analysis['total_groups']}")
    lines.append(f"  单 granule 组: {group_analysis['single_granule_groups']}")
    lines.append(f"  CRS 一致: {group_analysis['consistent_groups']}")
    lines.append(f"  CRS 不一致: {group_analysis['inconsistent_groups']}")

    # 不一致的时间组详情
    if group_analysis['inconsistent_details']:
        lines.append("\n## ⚠️ 5. CRS 不一致的时间组详情")
        lines.append("-" * 40)
        for item in group_analysis['inconsistent_details'][:5]:
            lines.append(f"\n  时间组: {item['group_id']}")
            lines.append(f"  UTM Zones: {item['utm_zones']}")
            lines.append(f"  Granule 数量: {item['granule_count']}")
            for d in item['details']:
                lines.append(f"    - {d['filename']}: zone={d['utm_zone']}, epsg={d['epsg']}")

    # 结论
    lines.append("\n## 6. 结论")
    lines.append("-" * 40)
    if group_analysis['inconsistent_groups'] > 0:
        lines.append("  ⚠️ 发现同一时间组内存在不同 UTM Zone 的情况！")
        lines.append("  这意味着代码需要在拼接前进行投影转换。")
    else:
        lines.append("  ✅ 所有时间组内的 CRS 一致，无需投影转换。")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='SWOT Raster 投影验证工具')
    parser.add_argument('--dir', '-d', type=str, required=True,
                        help='数据目录')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出报告文件路径（JSON格式）')
    parser.add_argument('--max-files', '-m', type=int, default=None,
                        help='最大检查文件数（用于测试）')
    parser.add_argument('--sample-groups', '-s', type=int, default=None,
                        help='抽样检查的时间组数量')

    args = parser.parse_args()

    # 分析 UTM zone 分布
    print("\n[1/2] 分析 UTM Zone 分布...")
    utm_analysis = analyze_utm_zones(args.dir, args.max_files)

    # 检查时间组一致性
    print("\n[2/2] 检查时间组 CRS 一致性...")
    all_files = scan_raster_files(args.dir)
    if args.max_files:
        all_files = all_files[:args.max_files]
    time_groups = group_granules_by_time(all_files)
    group_analysis = check_time_group_crs_consistency(time_groups, args.sample_groups)

    # 生成报告
    report = generate_report(utm_analysis, group_analysis)
    print(report)

    # 保存 JSON 结果
    if args.output:
        output_data = {
            'utm_analysis': {
                'total_files': utm_analysis['total_files'],
                'utm_zone_by_filename': dict(utm_analysis['utm_zone_by_filename']),
                'utm_zone_by_content': dict(utm_analysis['utm_zone_by_content']),
                'mismatch_count': len(utm_analysis['filename_content_mismatch'])
            },
            'group_analysis': {
                'total_groups': group_analysis['total_groups'],
                'consistent_groups': group_analysis['consistent_groups'],
                'inconsistent_groups': group_analysis['inconsistent_groups'],
                'single_granule_groups': group_analysis['single_granule_groups'],
                'inconsistent_details': group_analysis['inconsistent_details']
            }
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {args.output}")


if __name__ == '__main__':
    main()

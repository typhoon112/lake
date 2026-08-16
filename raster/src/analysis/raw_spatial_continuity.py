"""
分析原始 SWOT Raster 数据（未经 QC）的湖泊空间连续性

选取布南湖 (id=1) 作为样本，从原始 NetCDF 中裁切并分析:
1. 有效像元覆盖率和分布
2. 连通分量数和空间连续性指标
3. 与先验湖泊边界的重叠分析
4. C/PIC0 vs C/PIC2 vs D/PID0 版本对比
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_window, rasterize
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, mapping
from scipy import ndimage
import glob
import os
import sys

# 布南湖中心
TARGET_LON, TARGET_LAT = 90.84, 35.95
LAKE_ID = 1


def find_raw_files(data_dir, passes=['202', '217', '495', '508', '523'], max_per_version=3):
    """找到覆盖布南湖的原始 nc 文件"""
    files = sorted(glob.glob(os.path.join(data_dir, 'SWOT_*.nc')))
    results = {'PIC0': [], 'PIC2': [], 'PID0': [], 'PGC0': []}

    for f in files:
        base = os.path.basename(f)
        # 判断版本
        version = None
        for v in results:
            if f'_{v}_' in base:
                version = v
                break
        if not version:
            continue

        # 检查pass
        pass_match = False
        for p in passes:
            if f'_{p}_' in base:
                pass_match = True
                break
        if not pass_match:
            continue

        # 检查覆盖
        try:
            ds = xr.open_dataset(f)
            lon = ds['longitude'].values
            lat = ds['latitude'].values
            ds.close()
            if (np.nanmin(lon) <= TARGET_LON <= np.nanmax(lon) and
                np.nanmin(lat) <= TARGET_LAT <= np.nanmax(lat)):
                if len(results[version]) < max_per_version:
                    results[version].append(f)
        except:
            pass

    return results


def clip_to_bbox(nc_path, lake_bbox_utm, lake_geom_utm):
    """从原始 nc 裁切到湖泊 bbox，返回 wse 数组和元数据"""
    ds = xr.open_dataset(nc_path)

    # 获取空间参考
    crs_wkt = ds['crs'].attrs.get('spatial_ref', '')
    transform_str = ds['crs'].attrs.get('GeoTransform', '')

    # 获取 wse 数据
    wse = ds['wse'].values.astype(np.float32)
    lon = ds['longitude'].values
    lat = ds['latitude'].values

    # 构建 affine transform（100m 分辨率）
    ny, nx = wse.shape
    # 用经纬度网格中心计算
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

    # 获取 x/y 坐标（如果有的话）
    # SWOT raster 100m 产品通常有暗含的 x/y 坐标
    # 从 crs 属性获取 UTM 参数
    from pyproj import CRS, Transformer
    from rasterio.transform import from_origin

    crs_obj = CRS.from_wkt(crs_wkt) if crs_wkt else None
    if crs_obj is None:
        ds.close()
        return None

    # 用 pyproj 转换湖泊 bbox 到 nc 的 CRS
    transformer = Transformer.from_crs('EPSG:32646', crs_obj, always_xy=True)
    bbox = lake_bbox_utm.bounds
    # 转换四个角
    x_min, y_min = transformer.transform(bbox[0], bbox[1])
    x_max, y_max = transformer.transform(bbox[2], bbox[3])

    # 从 x/y 坐标构建网格
    # 100m 分辨率
    res = 100.0
    nc_x_min = float(ds['crs'].attrs.get('GeoTransform', '0 100 0 0 0 -100').split()[0])
    x_coords = np.arange(nx) * res + nc_x_min
    y_coords = np.arange(ny) * (-res) + abs(float(ds['crs'].attrs.get('GeoTransform', '0 100 0 0 0 -100').split()[3]))

    # 找裁切窗口
    col_min = max(0, int((x_min - x_coords[0]) / res))
    col_max = min(nx, int((x_max - x_coords[0]) / res) + 1)
    row_min = max(0, int((y_coords[0] - y_max) / res))
    row_max = min(ny, int((y_coords[0] - y_min) / res) + 1)

    if col_min >= col_max or row_min >= row_max:
        ds.close()
        return None

    wse_clip = wse[row_min:row_max, col_min:col_max]

    # 标记无效值 (-9999.0 或 NaN)
    wse_clip = np.where(wse_clip < -9990, np.nan, wse_clip)

    ds.close()

    return {
        'wse': wse_clip,
        'shape': wse_clip.shape,
        'nc_file': os.path.basename(nc_path),
    }


def analyze_spatial_continuity(wse_2d, label=""):
    """分析空间连续性"""
    valid_mask = ~np.isnan(wse_2d)
    valid = wse_2d[valid_mask]
    total = wse_2d.size

    if len(valid) == 0:
        print(f"  [{label}] 无有效数据!")
        return

    print(f"  [{label}] 有效像元: {len(valid)}/{total} ({len(valid)/total*100:.1f}%)")
    print(f"  [{label}] WSE: min={np.nanmin(valid):.2f}, max={np.nanmax(valid):.2f}, "
          f"mean={np.nanmean(valid):.2f}, std={np.nanstd(valid):.2f} m")

    # 连通分量
    labeled, num_features = ndimage.label(valid_mask)
    print(f"  [{label}] 连通分量数: {num_features}")

    sizes = ndimage.sum(valid_mask, labeled, range(1, num_features + 1))
    sizes = sorted(sizes, reverse=True)
    print(f"  [{label}] 最大分量: {sizes[0]:.0f} 像元 (占有效 {sizes[0]/len(valid)*100:.1f}%)")
    if len(sizes) > 1:
        print(f"  [{label}] 前5分量: {[f'{s:.0f}' for s in sizes[:5]]}")

    # 外接矩形填充率
    rows, cols = np.where(valid_mask)
    bbox_area = (cols.max() - cols.min() + 1) * (rows.max() - rows.min() + 1)
    print(f"  [{label}] 外接矩形填充率: {len(valid)/bbox_area*100:.1f}%")
    print(f"  [{label}] 空间连续性指标: {sizes[0]/len(valid)*100:.1f}%")

    # 空洞分析（最大分量内部）
    main_component = labeled == (np.argmax(sizes) + 1)
    # 在主分量外接矩形内，无效像元 = 空洞
    r1, r2 = rows.min(), rows.max()
    c1, c2 = cols.min(), cols.max()
    roi = valid_mask[r1:r2+1, c1:c2+1]
    roi_main = main_component[r1:r2+1, c1:c2+1]
    holes = roi_main & ~roi
    print(f"  [{label}] 主分量内空洞: {holes.sum()} 像元")

    return {
        'valid_ratio': len(valid)/total,
        'num_components': num_features,
        'max_component_ratio': sizes[0]/len(valid),
        'fill_ratio': len(valid)/bbox_area,
    }


def main():
    print("=" * 70)
    print("原始 SWOT Raster 数据空间连续性分析")
    print(f"目标: 布南湖 (id={LAKE_ID}, 中心 ~90.84°E, 35.95°N)")
    print("=" * 70)

    # 加载湖泊
    gdf = gpd.read_file('data/prior/QB58_716_4.shp')
    lake = gdf[gdf['id_unified'] == LAKE_ID].iloc[0]
    lake_geom = lake.geometry
    buf = lake_geom.buffer(500)
    lake_bbox = box(*buf.bounds)
    print(f"\n布南湖面积: {lake.geometry.area/1e6:.2f} km²")
    print(f"bbox: {lake.geometry.bounds}")

    # C版本
    print("\n" + "-" * 70)
    print("扫描 C 版本原始数据...")
    c_files = find_raw_files('data/raw/swot_raster_C', max_per_version=2)
    print(f"找到: " + ", ".join(f"{k}={len(v)}" for k, v in c_files.items()))

    # D版本
    print("\n扫描 D 版本原始数据...")
    d_files = find_raw_files('data/raw/swot_raster_D', max_per_version=2)
    print(f"找到: " + ", ".join(f"{k}={len(v)}" for k, v in d_files.items()))

    # 合并分析
    all_files = {}
    for v in ['PIC0', 'PIC2', 'PID0']:
        source = c_files if v != 'PID0' else d_files
        if source.get(v):
            all_files[v] = source[v]

    # 简化：直接从 nc 读 wse，用经纬度裁切
    print("\n" + "=" * 70)
    print("逐样本分析")
    print("=" * 70)

    for version, files in all_files.items():
        for fpath in files:
            fname = os.path.basename(fpath)
            print(f"\n{'─' * 60}")
            print(f"文件: {fname[:80]}")

            try:
                ds = xr.open_dataset(fpath)
                wse = ds['wse'].values.astype(np.float32)
                lon = ds['longitude'].values
                lat = ds['latitude'].values

                # 标记无效
                wse[wse < -9990] = np.nan

                # 用经纬度裁切到湖泊范围（扩大一点 buffer）
                lake_wgs84 = gdf[gdf['id_unified'] == LAKE_ID].to_crs(epsg=4326)
                b84 = lake_wgs84.iloc[0].geometry.buffer(0.02).bounds  # ~2km buffer

                mask_lon = (lon >= b84[0]) & (lon <= b84[2])
                mask_lat = (lat >= b84[1]) & (lat <= b84[3])
                row_mask = mask_lat.any(axis=1)
                col_mask = mask_lon.any(axis=0)

                r_start, r_end = np.where(row_mask)[0][[0, -1]]
                c_start, c_end = np.where(col_mask)[0][[0, -1]]

                wse_clip = wse[r_start:r_end+1, c_start:c_end+1]
                lon_clip = lon[r_start:r_end+1, c_start:c_end+1]
                lat_clip = lat[r_start:r_end+1, c_start:c_end+1]

                res = 100  # 100m
                print(f"  裁切窗口: {wse_clip.shape[1]}x{wse_clip.shape[0]} "
                      f"({wse_clip.shape[1]*res/1000:.1f}km x {wse_clip.shape[0]*res/1000:.1f}km)")

                analyze_spatial_continuity(wse_clip, version)
                ds.close()

            except Exception as e:
                print(f"  处理失败: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == '__main__':
    main()

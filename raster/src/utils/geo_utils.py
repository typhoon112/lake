"""
地理空间工具模块

提供坐标转换、空间操作等地理工具函数
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import box, mapping, shape
from typing import Union, Tuple, List, Optional
import pyproj
from rasterio.crs import CRS
from rasterio.warp import transform_bounds


# 常用坐标系统
CRS_WGS84 = "EPSG:4326"  # WGS84 地理坐标系
CRS_ALBERS_ASIA = "EPSG:102025"  # Albers Asia 等面积投影


def get_utm_zone_from_bounds(bounds: Tuple[float, float, float, float],
                              src_crs: str = CRS_WGS84) -> str:
    """
    根据边界获取对应的UTM Zone EPSG代码

    Args:
        bounds: (minx, miny, maxx, maxy) 边界坐标
        src_crs: 输入坐标的CRS

    Returns:
        UTM Zone的EPSG代码，如 "EPSG:32650" (北半球) 或 "EPSG:32750" (南半球)
    """
    # 计算中心经纬度
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    # 计算UTM Zone编号
    zone_number = int((center_lon + 180) / 6) + 1

    # 根据纬度确定北半球还是南半球
    if center_lat >= 0:
        epsg_code = 32600 + zone_number  # 北半球
    else:
        epsg_code = 32700 + zone_number  # 南半球

    return f"EPSG:{epsg_code}"


def transform_bounds_to_crs(bounds: Tuple[float, float, float, float],
                            src_crs: Union[str, CRS],
                            dst_crs: Union[str, CRS]) -> Tuple[float, float, float, float]:
    """
    将边界坐标从一个坐标系转换到另一个坐标系

    Args:
        bounds: (minx, miny, maxx, maxy) 源坐标系的边界
        src_crs: 源坐标系
        dst_crs: 目标坐标系

    Returns:
        转换后的边界坐标
    """
    if isinstance(src_crs, str):
        src_crs = CRS.from_string(src_crs)
    if isinstance(dst_crs, str):
        dst_crs = CRS.from_string(dst_crs)

    return transform_bounds(src_crs, dst_crs, *bounds)


def bounds_overlap(bounds1: Tuple[float, float, float, float],
                   bounds2: Tuple[float, float, float, float]) -> bool:
    """
    检查两个边界框是否重叠

    Args:
        bounds1: (minx, miny, maxx, maxy) 第一个边界
        bounds2: (minx, miny, maxx, maxy) 第二个边界

    Returns:
        是否重叠
    """
    return not (bounds1[2] < bounds2[0] or  # bounds1在bounds2左边
                bounds1[0] > bounds2[2] or  # bounds1在bounds2右边
                bounds1[3] < bounds2[1] or  # bounds1在bounds2下边
                bounds1[1] > bounds2[3])    # bounds1在bounds2上边


def gdf_to_bounds(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """
    获取GeoDataFrame的总边界（在WGS84坐标系下）

    Args:
        gdf: GeoDataFrame

    Returns:
        (minx, miny, maxx, maxy) WGS84坐标系的边界
    """
    if gdf.crs != CRS_WGS84:
        gdf_wgs84 = gdf.to_crs(CRS_WGS84)
    else:
        gdf_wgs84 = gdf

    return gdf_wgs84.total_bounds


def calculate_area_in_equal_area(gdf: gpd.GeoDataFrame,
                                  target_crs: str = CRS_ALBERS_ASIA) -> float:
    """
    在等面积投影下计算几何体的总面积

    Args:
        gdf: GeoDataFrame
        target_crs: 等面积投影坐标系

    Returns:
        总面积（平方米）
    """
    # 转换到等面积投影
    gdf_projected = gdf.to_crs(target_crs)

    # 计算面积
    return gdf_projected.geometry.area.sum()


def point_in_bounds(point: Tuple[float, float],
                    bounds: Tuple[float, float, float, float]) -> bool:
    """
    检查点是否在边界框内

    Args:
        point: (x, y) 点坐标
        bounds: (minx, miny, maxx, maxy) 边界框

    Returns:
        点是否在边界框内
    """
    return (bounds[0] <= point[0] <= bounds[2] and
            bounds[1] <= point[1] <= bounds[3])


def create_bbox_geometry(bounds: Tuple[float, float, float, float],
                         crs: str = CRS_WGS84) -> gpd.GeoDataFrame:
    """
    根据边界创建边界框GeoDataFrame

    Args:
        bounds: (minx, miny, maxx, maxy) 边界坐标
        crs: 坐标系

    Returns:
        包含边界框的GeoDataFrame
    """
    bbox = box(*bounds)
    gdf = gpd.GeoDataFrame(geometry=[bbox], crs=crs)
    return gdf


def parse_swot_utm_grid(grid_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    解析SWOT文件名中的UTM Grid信息

    Args:
        grid_name: 如 "UTM01C" 或 "UTM01W"

    Returns:
        (zone_number, hemisphere) 如 ("01", "C") 或 ("01", "W")
        C = Central, W = West (表示东西半球)
    """
    if grid_name.startswith("UTM") and len(grid_name) >= 5:
        zone = grid_name[3:5]  # 01
        hemisphere = grid_name[5]  # C or W
        return zone, hemisphere
    return None, None


def get_epsg_from_swot_grid(grid_name: str) -> Optional[str]:
    """
    从SWOT UTM Grid名称获取EPSG代码

    注意: SWOT的UTM grid编码与标准EPSG不完全对应
    需要根据实际数据的CRS信息确定

    Args:
        grid_name: 如 "UTM01C"

    Returns:
        EPSG代码，如果无法确定则返回None
    """
    # SWOT UTM Grid的前两位数字代表zone
    # C/W后缀表示该grid在zone的中心还是西侧
    # 实际EPSG需要从数据中读取
    zone, _ = parse_swot_utm_grid(grid_name)
    if zone:
        # 默认返回北半球的EPSG，实际使用时需要从数据中确认
        return f"EPSG:326{zone}"
    return None

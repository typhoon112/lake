"""
SWOT湖泊数据汇总脚本
输出每个湖泊的观测次数、时间范围、水位统计等信息
"""
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from glob import glob

# 配置路径
OUTPUT_DIR = "/data/lyf/Lyf_data/SWOT/SWOT_By_Lake/"
OBS_DIR = os.path.join(OUTPUT_DIR, "Obs_Lakes")
UNAS_DIR = os.path.join(OUTPUT_DIR, "Unassigned_Lakes")
OUTPUT_CSV = "/data/lyf/geo/Lake/SWOT数据处理/湖泊统计汇总.csv"


def process_lake_folder(lake_dir, lake_type):
    """处理单个湖泊文件夹"""
    shp_files = glob(os.path.join(lake_dir, "*.shp"))
    if not shp_files:
        return None

    shp_path = shp_files[0]
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        print(f"读取失败 {lake_dir}: {e}")
        return None

    # 提取湖泊ID
    lake_id = lake_dir.replace("Lake_", "")

    # 统计信息
    n_obs = len(gdf)
    time_range = f"{gdf['time_utc'].min()} ~ {gdf['time_utc'].max()}" if 'time_utc' in gdf.columns else "N/A"

    # 水位统计
    if 'wse' in gdf.columns:
        wse_min = gdf['wse'].min()
        wse_max = gdf['wse'].max()
        wse_mean = gdf['wse'].mean()
        wse_std = gdf['wse'].std() if n_obs > 1 else 0
    else:
        wse_min = wse_max = wse_mean = wse_std = np.nan

    # 水位不确定度统计
    if 'wse_u' in gdf.columns:
        wse_u_mean = gdf['wse_u'].mean()
        wse_u_max = gdf['wse_u'].max()
    else:
        wse_u_mean = wse_u_max = np.nan

    # 面积统计
    if 'area' in gdf.columns:
        area_min = gdf['area'].min()
        area_max = gdf['area'].max()
        area_mean = gdf['area'].mean()
    else:
        area_min = area_max = area_mean = np.nan

    # 几何类型
    geom_types = ', '.join(gdf.geom_type.unique())

    return {
        'lake_id': lake_id,
        'lake_type': lake_type,
        'n_obs': n_obs,
        'time_range': time_range,
        'wse_min_m': round(wse_min, 3) if not np.isnan(wse_min) else np.nan,
        'wse_max_m': round(wse_max, 3) if not np.isnan(wse_max) else np.nan,
        'wse_mean_m': round(wse_mean, 3) if not np.isnan(wse_mean) else np.nan,
        'wse_std_m': round(wse_std, 3) if not np.isnan(wse_std) else np.nan,
        'wse_u_mean_m': round(wse_u_mean, 4) if not np.isnan(wse_u_mean) else np.nan,
        'wse_u_max_m': round(wse_u_max, 4) if not np.isnan(wse_u_max) else np.nan,
        'area_min_km2': round(area_min, 4) if not np.isnan(area_min) else np.nan,
        'area_max_km2': round(area_max, 4) if not np.isnan(area_max) else np.nan,
        'area_mean_km2': round(area_mean, 4) if not np.isnan(area_mean) else np.nan,
        'geom_type': geom_types,
        'shp_file': os.path.basename(shp_path)
    }


def main():
    print("开始汇总湖泊数据...")

    all_lakes = []

    # 处理 Obs_Lakes
    if os.path.exists(OBS_DIR):
        lake_dirs = [d for d in os.listdir(OBS_DIR) if os.path.isdir(os.path.join(OBS_DIR, d))]
        print(f"处理 Obs_Lakes: {len(lake_dirs)} 个湖泊")
        for i, lake_dir in enumerate(lake_dirs):
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i+1}/{len(lake_dirs)}")
            result = process_lake_folder(os.path.join(OBS_DIR, lake_dir), "Obs")
            if result:
                all_lakes.append(result)

    # 处理 Unassigned_Lakes
    if os.path.exists(UNAS_DIR):
        lake_dirs = [d for d in os.listdir(UNAS_DIR) if os.path.isdir(os.path.join(UNAS_DIR, d))]
        print(f"处理 Unassigned_Lakes: {len(lake_dirs)} 个湖泊")
        for i, lake_dir in enumerate(lake_dirs):
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i+1}/{len(lake_dirs)}")
            result = process_lake_folder(os.path.join(UNAS_DIR, lake_dir), "Unassigned")
            if result:
                all_lakes.append(result)

    # 创建DataFrame并保存
    df = pd.DataFrame(all_lakes)
    df = df.sort_values('n_obs', ascending=False)

    # 保存CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n汇总完成！共 {len(df)} 个湖泊")
    print(f"结果已保存到: {OUTPUT_CSV}")

    # 输出统计摘要
    print("\n=== 数据可用性统计 ===")
    print(f"总湖泊数: {len(df)}")
    print(f"总观测次数: {df['n_obs'].sum()}")
    print(f"\n观测次数分布:")
    print(df['n_obs'].describe())
    print(f"\n单次观测湖泊数: {len(df[df['n_obs'] == 1])}")
    print(f"多次观测湖泊数: {len(df[df['n_obs'] > 1])}")
    print(f"10次以上观测: {len(df[df['n_obs'] > 10])}")

    # 水位不确定度分布
    print(f"\n水位不确定度分布:")
    valid_wse_u = df[df['wse_u_mean_m'].notna()]['wse_u_mean_m']
    print(f"  最小值: {valid_wse_u.min():.4f} m")
    print(f"  最大值: {valid_wse_u.max():.4f} m")
    print(f"  平均值: {valid_wse_u.mean():.4f} m")
    print(f"  中位数: {valid_wse_u.median():.4f} m")


if __name__ == "__main__":
    main()

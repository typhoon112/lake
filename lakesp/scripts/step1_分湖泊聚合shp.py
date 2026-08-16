import geopandas as gpd
import pandas as pd
import numpy as np
import os
from glob import glob
from shapely.ops import unary_union
import fiona
import logging
import warnings
from tqdm import tqdm  # 引入进度条库

# 忽略 Pandas 的一些未来版本警告
warnings.filterwarnings('ignore')

# --- 用户配置区 (已更新为你提供的路径) ---
INPUT_DIR = "/data/lyf/Lyf_data/SWOT/SWOT_Source"
OUTPUT_DIR = "/data/lyf/Lyf_data/SWOT/SWOT_By_Lake/"
LOG_FILE = "swot_processing_tqdm.log"

# 创建输出目录
obs_dir = os.path.join(OUTPUT_DIR, "Obs_Lakes")
unas_dir = os.path.join(OUTPUT_DIR, "Unassigned_Lakes")
for d in [obs_dir, unas_dir]:
    os.makedirs(d, exist_ok=True)

# 配置日志
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_weighted_mean(values, uncertainties):
    """
    官方建议的加权平均算法：基于不确定度(wse_u)的平方倒数加权
    """
    vals = np.array(values)
    uncs = np.array(uncertainties)

    # 防止除以0，给极小的 uncertainty 一个下限
    uncs = np.maximum(uncs, 1e-4)

    weights = 1.0 / (uncs ** 2)
    # 处理全0权重的情况
    if np.sum(weights) == 0:
        return np.mean(vals), 0

    weighted_mean = np.sum(vals * weights) / np.sum(weights)

    # 合成后的不确定度 (标准误差传播公式)
    combined_uncertainty = np.sqrt(1.0 / np.sum(weights))

    return weighted_mean, combined_uncertainty


def process_swot_with_tqdm():
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("-" * 50)

    # 1. 扫描文件
    print("正在扫描 .shp 文件...")
    all_shps = glob(os.path.join(INPUT_DIR, "**/*.shp"), recursive=True)

    if not all_shps:
        print("错误：未找到任何 .shp 文件，请检查路径！")
        return

    lake_catalog = {}

    # --- 阶段 1: 建立索引 (带进度条) ---
    print("Step 1/2: 建立文件索引与 ID 映射...")
    # 使用 tqdm 包装循环
    for shp_path in tqdm(all_shps, desc="Indexing Files", unit="file"):
        is_unassigned = "Unassigned" in os.path.basename(shp_path)

        try:
            with fiona.open(shp_path) as src:
                if len(src) == 0: continue

                # 预读第一个要素获取 ID
                first_feat = next(iter(src))
                raw_lid = first_feat['properties'].get('lake_id')

                if raw_lid is None: continue

                # 处理 lake_id 可能包含多个ID的情况（用分号分隔），只取第一个
                if ';' in str(raw_lid):
                    raw_lid = str(raw_lid).split(';')[0]

                # 构造唯一 ID (区分 Unassigned)
                unique_lid = f"Unas_{raw_lid}" if is_unassigned else str(raw_lid)

                if unique_lid not in lake_catalog:
                    lake_catalog[unique_lid] = {"files": [], "type": "Unassigned" if is_unassigned else "Obs"}

                lake_catalog[unique_lid]["files"].append(shp_path)

        except Exception as e:
            logging.error(f"索引失败: {shp_path} - {e}")

    total_lakes = len(lake_catalog)
    print(f"索引完成。共发现 {total_lakes} 个独立湖泊对象。")

    # --- 阶段 2: 逐湖处理 (带进度条) ---
    print("Step 2/2: 执行空间聚合与时序重构...")

    # 使用 tqdm 包装字典项
    for lid, info in tqdm(lake_catalog.items(), desc="Processing Lakes", total=total_lakes, unit="lake"):
        file_list = info["files"]
        lake_type = info["type"]
        target_dir = unas_dir if lake_type == "Unassigned" else obs_dir

        # 内存保护：跳过碎片过多的大型异常聚合
        if len(file_list) > 600:
            logging.warning(f"湖泊 {lid} 关联文件过多 ({len(file_list)})，已跳过")
            continue

        try:
            dfs = []
            for f in file_list:
                try:
                    # 读取 shapefile
                    gdf = gpd.read_file(f, ignore_geometry=False)

                    # --- 字段鲁棒性处理 (解决 KeyError: 'area') ---
                    # 1. 处理面积字段
                    if 'area_total' in gdf.columns:
                        gdf['proc_area'] = gdf['area_total']
                        gdf['proc_area_u'] = gdf['area_tot_u']
                    elif 'area' in gdf.columns:
                        gdf['proc_area'] = gdf['area']
                        gdf['proc_area_u'] = gdf['area_u'] if 'area_u' in gdf.columns else 0.1
                    elif 'area_calc' in gdf.columns:  # 处理部分 Unassigned 变体
                        gdf['proc_area'] = gdf['area_calc']
                        gdf['proc_area_u'] = 0.1
                    else:
                        gdf['proc_area'] = 0
                        gdf['proc_area_u'] = 0

                    # 2. 确保水位字段存在
                    if 'wse' not in gdf.columns:
                        continue
                    if 'wse_u' not in gdf.columns:
                        gdf['wse_u'] = 0.1  # 给个默认不确定度防止除0

                    # 3. ID 匹配 - 处理 lake_id 可能包含多个ID的情况
                    search_id = lid.replace("Unas_", "") if lake_type == "Unassigned" else lid
                    # 将 lake_id 转为字符串并处理分号情况
                    gdf['lake_id_str'] = gdf['lake_id'].astype(str).str.split(';').str[0]
                    # 匹配第一个ID
                    subset = gdf[gdf['lake_id_str'] == str(search_id)].copy()

                    if not subset.empty:
                        dfs.append(subset)

                except Exception as read_err:
                    logging.warning(f"读取文件失败 {f}: {read_err}")

            if not dfs: continue

            full_df = pd.concat(dfs, ignore_index=True)

            # --- 科学过滤 ---
            # 过滤 Quality Flag: 0=未评估, 1=好, 2=可疑, 3=坏
            # 保留质量为 0, 1, 2 的记录（未评估、好、可疑）
            if 'quality_f' in full_df.columns:
                full_df = full_df[full_df['quality_f'] <= 2]

            # 过滤冰期 - 使用 ice_clim_f（气候冰期标志）
            # ice_clim_f: 0=无冰, 1=有冰, -999=缺失值
            if 'ice_clim_f' in full_df.columns:
                full_df = full_df[(full_df['ice_clim_f'] == 0) | (full_df['ice_clim_f'] == -999)]

            if full_df.empty: continue

            # --- 聚合计算 ---
            time_series = []

            for t_str, group in full_df.groupby('time_str'):
                # 预处理：过滤空几何和无效几何
                valid_geoms = []
                for g in group.geometry:
                    # 先检查是否为 None
                    if g is None:
                        continue
                    # 检查是否有 is_empty 属性
                    try:
                        if g.is_empty:
                            continue
                    except AttributeError:
                        continue
                    # 检查是否有效
                    try:
                        if not g.is_valid:
                            continue
                    except AttributeError:
                        continue
                    valid_geoms.append(g)

                if not valid_geoms:
                    continue

                # 几何融合
                try:
                    combined_geom = unary_union(valid_geoms)
                except Exception as e:
                    logging.warning(f"几何融合失败 {lid} @ {t_str}: {e}")
                    continue

                # 处理 Geometry Collection 类型（shapefile不支持）
                if combined_geom is not None and combined_geom.geom_type == 'GeometryCollection':
                    # 提取其中的有效polygon/multipolygon部分
                    new_geoms = []
                    for g in combined_geom.geoms:
                        if g is None:
                            continue
                        if not hasattr(g, 'geom_type'):
                            continue
                        if g.geom_type not in ('Polygon', 'MultiPolygon'):
                            continue
                        # 检查 is_empty
                        try:
                            if g.is_empty:
                                continue
                        except (AttributeError, TypeError):
                            continue
                        new_geoms.append(g)
                    if new_geoms:
                        combined_geom = unary_union(new_geoms)
                    else:
                        continue

                # 确保结果是有效的 Polygon 或 MultiPolygon
                if combined_geom is None:
                    continue
                # 只有非 None 时才检查 is_empty
                if hasattr(combined_geom, 'is_empty') and combined_geom.is_empty:
                    continue

                # 确保 geom_type 属性存在
                if not hasattr(combined_geom, 'geom_type'):
                    continue

                # 转换为 MultiPolygon 以确保兼容性
                if combined_geom.geom_type == 'Polygon':
                    from shapely.geometry import MultiPolygon
                    combined_geom = MultiPolygon([combined_geom])

                # 属性计算
                total_area = group['proc_area'].sum()
                total_area_u = np.sqrt((group['proc_area_u'] ** 2).sum())
                avg_wse, avg_wse_u = get_weighted_mean(group['wse'], group['wse_u'])

                # 记录 Layover (地形影响)
                avg_layover = group['layovr_val'].mean() if 'layovr_val' in group.columns else 0

                record = {
                    'lake_id': lid,
                    'time_utc': t_str,
                    'wse': avg_wse,
                    'wse_u': avg_wse_u,
                    'area': total_area,
                    'area_u': total_area_u,
                    'layover_avg': avg_layover,
                    'geometry': combined_geom
                }
                time_series.append(record)

            # 保存结果
            if time_series:
                res_gdf = gpd.GeoDataFrame(time_series, crs=full_df.crs)
                res_gdf = res_gdf.sort_values('time_utc')

                # 处理 lake_id 中可能包含的分号，替换为下划线以避免文件系统问题
                safe_lid = lid.replace(';', '_')

                # 每个湖泊一个文件夹
                lake_folder = os.path.join(target_dir, f"Lake_{safe_lid}")
                os.makedirs(lake_folder, exist_ok=True)
                out_name = os.path.join(lake_folder, f"Lake_{safe_lid}.shp")
                res_gdf.to_file(out_name)

        except Exception as e:
            logging.error(f"处理湖泊 {lid} 失败: {e}")
            continue

    print("-" * 50)
    print(f"全部完成！")
    print(f"日志文件: {LOG_FILE}")
    print(f"结果路径: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_swot_with_tqdm()
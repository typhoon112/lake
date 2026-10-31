"""
面积-水位 (A-h) 曲线拟合模块

为每个湖泊拟合 area-WSE 曲线（hypsometric curve），支持：
1. 箱线图异常值剔除
2. 不确定度加权多类型曲线拟合（5 种函数）
3. BIC 模型选择 + 单调性验证
4. 质量分级与诊断图输出
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..config.settings import CURVE_FIT_CONFIG, PROJECT_ROOT

# 抑制拟合警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*covariance.*')


# ============================================================
# 候选函数定义
# ============================================================

def _linear(h, a, b):
    return a * h + b

def _quadratic(h, a, b, c):
    return a * h**2 + b * h + c

def _power(h, a, b, h0):
    return a * np.maximum(h - h0, 1e-6) ** b

def _exponential(h, a, b, h0):
    return a * np.exp(b * (h - h0))

def _logarithmic(h, a, b, h0):
    return a * np.log(np.maximum(h - h0, 1e-6)) + b


# 函数注册表：名称 → (函数, 参数数量, 参数初始值生成函数)
FIT_FUNCTIONS = {
    'linear':       (_linear,       2, None),
    'quadratic':    (_quadratic,    3, None),
    'power':        (_power,        3, 'shifted'),
    'exponential':  (_exponential,  3, 'shifted'),
    'logarithmic':  (_logarithmic,  3, 'shifted'),
}


def _generate_p0(func_name, n_params, wse, area):
    """根据数据自动生成初始参数估计"""
    h_min, h_max = wse.min(), wse.max()
    h_range = h_max - h_min if h_max > h_min else 1.0
    a_min, a_max = area.min(), area.max()
    a_range = a_max - a_min if a_max > a_min else 1.0

    if func_name == 'linear':
        slope = a_range / h_range if h_range > 0 else 1.0
        return [slope, a_min - slope * h_min]

    elif func_name == 'quadratic':
        slope = a_range / h_range if h_range > 0 else 1.0
        return [0.0, slope, a_min]

    elif func_name in ('power', 'exponential', 'logarithmic'):
        # shifted 类函数：h0 取比最小 WSE 略小
        h0 = h_min - 0.1 * h_range
        slope = a_range / h_range if h_range > 0 else 1.0
        return [slope, 1.0, h0]

    return None


# ============================================================
# 数据清洗
# ============================================================

def clean_outliers_iqr(area: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
    """
    箱线图异常值剔除（对面积列）

    Args:
        area: 面积数组
        iqr_factor: IQR 倍数

    Returns:
        布尔 mask，True 表示保留
    """
    q1 = np.percentile(area, 25)
    q3 = np.percentile(area, 75)
    iqr = q3 - q1

    if iqr == 0:
        # 所有面积相同，无异常值
        return np.ones(len(area), dtype=bool)

    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    return (area >= lower) & (area <= upper)


# ============================================================
# 单个湖泊拟合
# ============================================================

def _check_monotonic(func, params, wse_min, wse_max, n_samples=200):
    """检查拟合函数在数据范围内是否单调非降"""
    h_test = np.linspace(wse_min, wse_max, n_samples)
    try:
        values = func(h_test, *params)
        return np.all(np.diff(values) >= -1e-10)
    except Exception:
        return False


def _calc_bic(residuals, n_params):
    """计算 BIC"""
    n = len(residuals)
    if n == 0:
        return np.inf
    rss = np.sum(residuals**2)
    if rss <= 0:
        return -np.inf
    return n * np.log(rss / n) + n_params * np.log(n)


def _calc_r_squared(actual, predicted):
    """计算 R²"""
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_single_lake(wse: np.ndarray,
                     area: np.ndarray,
                     wse_uncert: np.ndarray,
                     config: Optional[Dict] = None) -> Dict:
    """
    对单个湖泊执行多类型曲线拟合

    Args:
        wse: 水位数组（已清洗）
        area: 面积数组（已清洗）
        wse_uncert: 水位不确定度数组（已清洗）
        config: 配置参数

    Returns:
        拟合结果字典
    """
    cfg = config or CURVE_FIT_CONFIG
    n = len(wse)
    n_curve_samples = cfg.get('n_curve_samples', 200)

    results_all = []
    best_result = None
    best_bic = np.inf

    for func_name, (func, n_params, p0_type) in FIT_FUNCTIONS.items():
        try:
            # 初始参数
            p0 = _generate_p0(func_name, n_params, wse, area)

            # 准备不确定度权重
            sigma = wse_uncert.copy()
            # 零或无效的 uncert 用中位数替代
            invalid_sigma = (sigma <= 0) | np.isnan(sigma)
            if invalid_sigma.any() and not invalid_sigma.all():
                median_sigma = np.median(sigma[~invalid_sigma])
                sigma[invalid_sigma] = median_sigma
            elif invalid_sigma.all():
                sigma = np.ones(n)

            # 拟合
            popt, pcov = curve_fit(
                func, wse, area,
                p0=p0,
                sigma=sigma,
                absolute_sigma=True,
                maxfev=10000,
            )

            # 残差与指标
            predicted = func(wse, *popt)
            residuals = area - predicted
            r_squared = _calc_r_squared(area, predicted)
            bic = _calc_bic(residuals, n_params)
            rmse = np.sqrt(np.mean(residuals**2))

            # 参数标准误差
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(n_params, np.nan)

            # 单调性检验
            monotonic = _check_monotonic(func, popt, wse.min(), wse.max(), n_curve_samples)

            result = {
                'function_type': func_name,
                'params': popt.tolist(),
                'param_errors': perr.tolist(),
                'r_squared': float(r_squared),
                'bic': float(bic),
                'rmse': float(rmse),
                'monotonic': bool(monotonic),
            }
            results_all.append(result)

            # 选最佳（BIC 最小且单调）
            if monotonic and bic < best_bic:
                best_bic = bic
                best_result = result

        except Exception:
            results_all.append({
                'function_type': func_name,
                'params': None,
                'param_errors': None,
                'r_squared': np.nan,
                'bic': np.inf,
                'rmse': np.nan,
                'monotonic': False,
            })

    # 如果没有单调的拟合结果，选 BIC 最小的（不要求单调）
    if best_result is None and results_all:
        valid_results = [r for r in results_all if r['params'] is not None]
        if valid_results:
            best_result = min(valid_results, key=lambda r: r['bic'])

    return {
        'best': best_result,
        'all_candidates': results_all,
    }


def assess_quality(result: Dict, n_cleaned: int, wse_range: float) -> str:
    """
    质量分级

    Returns:
        'reliable' | 'acceptable' | 'unreliable'
    """
    if result is None or result.get('params') is None:
        return 'unreliable'

    r2 = result.get('r_squared', 0)
    mono = result.get('monotonic', False)

    if r2 > 0.7 and n_cleaned >= 5 and wse_range > 1.0 and mono:
        return 'reliable'
    elif r2 > 0.5 and n_cleaned >= 3 and mono:
        return 'acceptable'
    else:
        return 'unreliable'


# ============================================================
# 诊断图
# ============================================================

def plot_diagnostic(lake_id, wse_orig, area_orig, wse_clean, area_clean,
                    wse_uncert_clean, fit_result, output_dir):
    """输出单个湖泊的诊断图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    # 原始数据（灰色）
    ax.scatter(wse_orig, area_orig / 1e6, c='lightgray', s=20, alpha=0.5,
               label='Raw', zorder=1)

    # 清洗后数据（蓝色，带不确定度误差棒）
    # 计算显示用误差棒：保证至少占 WSE 范围的 1.5%，否则看不见
    wse_span = wse_clean.max() - wse_clean.min()
    wse_span = wse_span if wse_span > 0 else 1.0
    min_visible_err = wse_span * 0.015
    display_err = np.maximum(wse_uncert_clean, min_visible_err)

    ax.errorbar(wse_clean, area_clean / 1e6,
                xerr=display_err, fmt='o', color='#2196F3',
                markersize=5, linewidth=1.2, capsize=4, capthick=1.2,
                alpha=0.8,
                label=f'Cleaned (n={len(wse_clean)})', zorder=2)

    # 拟合曲线
    if fit_result and fit_result.get('params') is not None:
        func_name = fit_result['function_type']
        func, _, _ = FIT_FUNCTIONS[func_name]
        params = fit_result['params']

        h_range = np.linspace(wse_clean.min(), wse_clean.max(), 300)
        try:
            fitted = func(h_range, *params)
            ax.plot(h_range, fitted / 1e6, 'r-', linewidth=2,
                    label=f'{func_name} fit', zorder=3)
        except Exception:
            pass

        # 标注
        r2 = fit_result.get('r_squared', np.nan)
        bic = fit_result.get('bic', np.nan)
        mono = 'Yes' if fit_result.get('monotonic') else 'No'
        quality = fit_result.get('_quality', 'N/A')
        info_text = (f'R² = {r2:.3f}\n'
                     f'BIC = {bic:.1f}\n'
                     f'Monotonic: {mono}\n'
                     f'Quality: {quality}')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('WSE (m)', fontsize=12)
    ax.set_ylabel('Water Area (km²)', fontsize=12)
    ax.set_title(f'Lake {lake_id} — Area-WSE Curve', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f'lake_{lake_id}_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 批量处理入口
# ============================================================

class AreaWSECurveFitter:
    """面积-水位曲线批量拟合器"""

    def __init__(self, csv_path: Optional[str] = None, config: Optional[Dict] = None):
        self.csv_path = csv_path or str(PROJECT_ROOT / 'output' / 'summary' / 'lake_timeseries.csv')
        self.config = config or CURVE_FIT_CONFIG
        self.output_dir = Path(PROJECT_ROOT) / 'output' / 'summary'

    def run(self):
        """执行全部湖泊的拟合"""
        # 读取数据
        df = pd.read_csv(self.csv_path)

        # 过滤有效 WSE
        valid = df[(df['wse'] != -9999.0) & (df['wse'].notna()) & (df['wse'] > 0)].copy()
        print(f"读取 {len(valid)} 条有效记录，覆盖 {valid['lake_id'].nunique()} 个湖泊")

        lake_ids = sorted(valid['lake_id'].unique())
        summary_rows = []
        detail_results = {}

        for lake_id in lake_ids:
            lake = valid[valid['lake_id'] == lake_id].copy().reset_index(drop=True)
            result = self._process_lake(lake_id, lake)
            summary_rows.append(result['summary'])
            detail_results[str(lake_id)] = result['detail']

        # 导出
        self._export_results(summary_rows, detail_results)
        self._print_summary(summary_rows)

    def _process_lake(self, lake_id, lake_df: pd.DataFrame) -> Dict:
        """处理单个湖泊"""
        n_orig = len(lake_df)
        wse = lake_df['wse'].values.astype(np.float64)
        area = lake_df['water_area'].values.astype(np.float64)
        wse_uncert = lake_df['wse_uncert'].values.astype(np.float64)

        # 保存原始数据用于绘图
        wse_orig = wse.copy()
        area_orig = area.copy()

        # 箱线图异常值剔除
        iqr_factor = self.config.get('iqr_factor', 1.5)
        mask = clean_outliers_iqr(area, iqr_factor)
        wse = wse[mask]
        area = area[mask]
        wse_uncert = wse_uncert[mask]
        n_cleaned = len(wse)

        # 检查最少数据点
        min_pts = self.config.get('min_cleaned_points', 3)
        if n_cleaned < min_pts:
            return {
                'summary': {
                    'lake_id': lake_id,
                    'n_obs': n_orig,
                    'n_cleaned': n_cleaned,
                    'function_type': 'none',
                    'params_json': '{}',
                    'r_squared': np.nan,
                    'bic': np.nan,
                    'rmse': np.nan,
                    'wse_range': 0.0,
                    'area_range_km2': 0.0,
                    'quality': 'unreliable',
                },
                'detail': {
                    'n_obs': n_orig,
                    'n_cleaned': n_cleaned,
                    'retention_rate': n_cleaned / n_orig * 100 if n_orig > 0 else 0,
                    'wse_range': 0.0,
                    'area_range_km2': 0.0,
                    'best_fit': None,
                    'all_candidates': [],
                },
            }

        # 拟合
        wse_range = float(wse.max() - wse.min())
        area_range = float((area.max() - area.min()) / 1e6)
        fit_result = fit_single_lake(wse, area, wse_uncert, self.config)

        # 质量评估
        best = fit_result['best']
        quality = assess_quality(best, n_cleaned, wse_range)

        # 标记质量（用于绘图）
        if best:
            best['_quality'] = quality

        # 诊断图
        if self.config.get('plot_diagnostics', True):
            plot_dir = self.output_dir / 'curve_plots'
            try:
                plot_diagnostic(
                    lake_id, wse_orig, area_orig,
                    wse, area, wse_uncert,
                    best, plot_dir,
                )
            except Exception as e:
                print(f"  Warning: 湖泊 {lake_id} 诊断图失败: {e}")

        # 汇总行
        func_type = best['function_type'] if best else 'none'
        params_json = json.dumps(best['params']) if best and best.get('params') else '{}'
        r2 = best['r_squared'] if best else np.nan
        bic = best['bic'] if best else np.nan
        rmse = best['rmse'] if best else np.nan

        return {
            'summary': {
                'lake_id': lake_id,
                'n_obs': n_orig,
                'n_cleaned': n_cleaned,
                'function_type': func_type,
                'params_json': params_json,
                'r_squared': r2,
                'bic': bic,
                'rmse': rmse,
                'wse_range': wse_range,
                'area_range_km2': area_range,
                'quality': quality,
            },
            'detail': {
                'n_obs': n_orig,
                'n_cleaned': n_cleaned,
                'retention_rate': n_cleaned / n_orig * 100,
                'wse_range': wse_range,
                'area_range_km2': area_range,
                'best_fit': best,
                'all_candidates': fit_result['all_candidates'],
            },
        }

    def _export_results(self, summary_rows: List[Dict], detail_results: Dict):
        """导出 CSV 和 JSON"""
        # CSV
        csv_path = self.output_dir / 'area_wse_curves.csv'
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(csv_path, index=False)
        print(f"CSV 汇总: {csv_path}")

        # JSON（去掉不可序列化的 _quality 标记）
        json_path = self.output_dir / 'area_wse_curves.json'
        export_details = {}
        for lid, detail in detail_results.items():
            d = dict(detail)
            if d.get('best_fit') and '_quality' in d['best_fit']:
                del d['best_fit']['_quality']
            export_details[lid] = d

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_details, f, indent=2, ensure_ascii=False, default=str)
        print(f"JSON 详细: {json_path}")

    def _print_summary(self, summary_rows: List[Dict]):
        """打印汇总统计"""
        df = pd.DataFrame(summary_rows)
        print(f"\n{'='*50}")
        print(f"面积-水位曲线拟合完成")
        print(f"{'='*50}")
        print(f"湖泊总数: {len(df)}")
        print(f"质量分级:")
        for q in ['reliable', 'acceptable', 'unreliable']:
            n = (df['quality'] == q).sum()
            print(f"  {q}: {n}")
        print(f"拟合函数分布:")
        print(df['function_type'].value_counts().to_string())


# ============================================================
# CLI 入口
# ============================================================

def main():
    fitter = AreaWSECurveFitter()
    fitter.run()


if __name__ == '__main__':
    main()

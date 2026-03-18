# coding: utf-8
"""
Reporting Module - 报告生成模块

合并自 html_reporter.py 和 reporter.py，提供完整的回测报告生成功能。

功能分层:
    P0:
        - 生成 HTML summary 报告
        - 包含 run summary, config summary, key metrics
        - equity curve, drawdown curve, trade summary
        - final holdout 结果放在最显眼位置
        - 保存回测输出到 Excel (trades, equity curve, config, risk metrics)

    P1:
        - monthly return table
        - yearly return table
        - baseline comparison
        - calibration section
        - feature importance section
        - execution stats
        - 自动输出图片并嵌入 HTML

    P2:
        - Dashboard 风格报告
        - 多 run 对比
        - Leaderboard 页面
        - 研究归档索引页
        - Regime 分析 (市场状态分层)

主要 API:
    - save_run_outputs(): 保存单次回测的所有输出 (Excel + HTML)
    - generate_multi_run_report(): 生成多 run 对比报告和索引页
    - save_html_report(): 保存 HTML 报告到文件
    - generate_leaderboard_html(): 生成排行榜 HTML
    - generate_index_html(): 生成研究归档索引页

使用示例:
    >>> from reporting import save_run_outputs, generate_multi_run_report
    >>> excel_path, config_path, run_id = save_run_outputs(
    ...     output_dir="output",
    ...     df_clean=df,
    ...     log_notes=notes,
    ...     trades=trades_df,
    ...     equity_curve=equity_df,
    ...     config=config_dict,
    ...     feature_meta=feature_meta_dict,
    ...     search_log=search_log_df,
    ... )
    >>> leaderboard_path = generate_multi_run_report(output_dir="output")
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from constants import (
    MIN_CLOSED_TRADES_PER_FOLD,
    TARGET_CLOSED_TRADES_PER_FOLD,
    LAMBDA_TRADE_PENALTY,
)
from evaluation import calculate_risk_metrics, format_metrics_summary, calculate_regime_metrics, calculate_trade_distribution, RegimeDetector, compute_regime_metrics, create_regime_visualization


# =============================================================================
# HTML 报告生成模块 (原 html_reporter.py)
# =============================================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _save_plot_to_base64(fig, format='png', dpi=100):
    """将 matplotlib 图表保存为 base64 编码。"""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64


def _create_equity_curve_plot(equity_curve: pd.DataFrame, initial_cash: float) -> Optional[str]:
    """创建净值曲线图。"""
    if not MATPLOTLIB_AVAILABLE or equity_curve.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    if 'total_equity' in equity_curve.columns:
        ax.plot(equity_curve.index, equity_curve['total_equity'], label='Strategy', linewidth=1.5)

    if 'bnh_equity' in equity_curve.columns:
        ax.plot(equity_curve.index, equity_curve['bnh_equity'], label='Buy & Hold', linewidth=1, alpha=0.7)

    ax.axhline(y=initial_cash, color='gray', linestyle='--', alpha=0.5, label='Initial Cash')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title('Equity Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return _save_plot_to_base64(fig)


def _create_drawdown_plot(equity_curve: pd.DataFrame) -> Optional[str]:
    """创建回撤曲线图。"""
    if not MATPLOTLIB_AVAILABLE or equity_curve.empty:
        return None

    if 'total_equity' not in equity_curve.columns:
        return None

    equity = equity_curve['total_equity'].values
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(equity_curve.index, drawdown, 0, alpha=0.3, color='red')
    ax.plot(equity_curve.index, drawdown, color='red', linewidth=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Curve')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return _save_plot_to_base64(fig)


def _create_monthly_returns_heatmap(monthly_returns: List[float]) -> Optional[str]:
    """创建月度收益热力图。"""
    if not MATPLOTLIB_AVAILABLE or not monthly_returns:
        return None

    fig, ax = plt.subplots(figsize=(12, 3))

    returns_array = np.array(monthly_returns).reshape(1, -1)
    im = ax.imshow(returns_array, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

    ax.set_yticks([])
    ax.set_xticks(range(len(monthly_returns)))
    ax.set_xticklabels([f'{i+1}' for i in range(len(monthly_returns))])
    ax.set_xlabel('Month')

    plt.colorbar(im, ax=ax, label='Return (%)')
    ax.set_title('Monthly Returns (%)')

    plt.tight_layout()
    return _save_plot_to_base64(fig)


def _create_trade_distribution_plot(trades: pd.DataFrame) -> Optional[str]:
    """创建交易分布图。"""
    if not MATPLOTLIB_AVAILABLE or trades is None or trades.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if 'pnl' in trades.columns:
        trades_pnl = trades['pnl'].dropna()
        if len(trades_pnl) > 0:
            axes[0].hist(trades_pnl, bins=30, edgecolor='black', alpha=0.7)
            axes[0].axvline(x=0, color='red', linestyle='--')
            axes[0].set_xlabel('PnL')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('PnL Distribution')
            axes[0].grid(True, alpha=0.3)

    if 'holding_days' in trades.columns:
        holding_days = trades['holding_days'].dropna()
        if len(holding_days) > 0:
            axes[1].hist(holding_days, bins=20, edgecolor='black', alpha=0.7, color='orange')
            axes[1].set_xlabel('Holding Days')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Holding Days Distribution')
            axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return _save_plot_to_base64(fig)


def _create_feature_importance_plot(importance_data: Dict[str, Any]) -> Optional[str]:
    """创建特征重要性图。"""
    if not MATPLOTLIB_AVAILABLE:
        return None

    ranking = importance_data.get('ranking', [])
    if not ranking:
        return None

    top_n = min(20, len(ranking))
    features = [r['feature'] for r in ranking[:top_n]][::-1]
    values = [r['importance'] for r in ranking[:top_n]][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(features, values, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return _save_plot_to_base64(fig)


def _create_calibration_plot(calibration_data: Dict[str, Any]) -> Optional[str]:
    """创建校准曲线图。"""
    if not MATPLOTLIB_AVAILABLE:
        return None

    curve = calibration_data.get('calibration_curve', {})
    if not curve:
        return None

    bin_centers = curve.get('bin_centers', [])
    bin_true_fractions = curve.get('bin_true_fractions', [])

    if not bin_centers:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.plot(bin_centers, bin_true_fractions, 'o-', label='Model calibration', markersize=8)

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    return _save_plot_to_base64(fig)


def _format_metric(value: float, metric_type: str = 'float') -> str:
    """格式化指标值。"""
    if metric_type == 'percent':
        return f"{value:+.2f}%" if value is not None else "N/A"
    elif metric_type == 'currency':
        return f"{value:,.2f}" if value is not None else "N/A"
    elif metric_type == 'int':
        return f"{int(value)}" if value is not None else "N/A"
    else:
        return f"{value:.4f}" if value is not None else "N/A"


def _build_config_section(config: Dict[str, Any]) -> str:
    """构建配置部分。"""
    sections = []

    for section in ['feature_config', 'model_config', 'trade_config', 'calibration_config']:
        if section in config:
            items = config[section]
            rows = []
            for k, v in items.items():
                rows.append(f'<div class="config-item"><div class="label">{k}</div><div class="value">{v}</div></div>')

            if rows:
                sections.append(f"""
                <h4>{section.replace('_', ' ').title()}</h4>
                <div class="config-grid">{''.join(rows)}</div>
                """)

    return ''.join(sections) if sections else '<p>No configuration details available.</p>'


def _build_metrics_section(metrics: Dict[str, float], benchmark_return: Optional[float], trade_count: int, win_rate: float) -> str:
    """构建指标部分（已在上层整合）。"""
    return ""


def _build_plots_section(plots: Dict[str, Optional[str]]) -> str:
    """构建图表部分。"""
    sections = []

    if plots.get('equity'):
        sections.append(f"""
        <div class="section">
            <h3>📉 Equity Curve</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{plots['equity']}" alt="Equity Curve">
            </div>
        </div>
        """)

    if plots.get('drawdown'):
        sections.append(f"""
        <div class="section">
            <h3>📉 Drawdown Curve</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{plots['drawdown']}" alt="Drawdown Curve">
            </div>
        </div>
        """)

    if plots.get('trade_dist'):
        sections.append(f"""
        <div class="section">
            <h3>📊 Trade Distribution</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{plots['trade_dist']}" alt="Trade Distribution">
            </div>
        </div>
        """)

    if plots.get('feature_importance'):
        sections.append(f"""
        <div class="section">
            <h3>🎯 Feature Importance</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{plots['feature_importance']}" alt="Feature Importance">
            </div>
        </div>
        """)

    if plots.get('calibration'):
        sections.append(f"""
        <div class="section">
            <h3>🎯 Calibration Curve</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{plots['calibration']}" alt="Calibration Curve">
            </div>
        </div>
        """)

    return ''.join(sections)


def _build_tables_section(
    trades: pd.DataFrame,
    monthly_returns: Optional[List[float]],
    yearly_returns: Optional[List[float]],
    feature_importance: Optional[Dict[str, Any]],
    feature_usage: Optional[Dict[str, Any]],
    calibration_data: Optional[Dict[str, Any]],
) -> str:
    """构建表格部分。"""
    sections = []

    if monthly_returns:
        rows = []
        for i, ret in enumerate(monthly_returns):
            cls = 'positive' if ret > 0 else 'negative'
            rows.append(f'<tr><td>Month {i+1}</td><td class="{cls}">{ret:+.2f}%</td></tr>')

        sections.append(f"""
        <div class="section">
            <h3>📅 Monthly Returns</h3>
            <table><tr><th>Month</th><th>Return</th></tr>{''.join(rows)}</table>
        </div>
        """)

    if yearly_returns:
        rows = []
        for i, ret in enumerate(yearly_returns):
            cls = 'positive' if ret > 0 else 'negative'
            rows.append(f'<tr><td>Year {i+1}</td><td class="{cls}">{ret:+.2f}%</td></tr>')

        sections.append(f"""
        <div class="section">
            <h3>📅 Yearly Returns</h3>
            <table><tr><th>Year</th><th>Return</th></tr>{''.join(rows)}</table>
        </div>
        """)

    if feature_importance and feature_importance.get('ranking'):
        ranking = feature_importance['ranking'][:15]
        rows = []
        for item in ranking:
            rows.append(f"<tr><td>{item['rank']}</td><td>{item['feature']}</td><td>{item['importance']:.6f}</td></tr>")

        sections.append(f"""
        <div class="section">
            <h3>🎯 Top Features</h3>
            <table><tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>{''.join(rows)}</table>
        </div>
        """)

    if calibration_data:
        brier_raw = calibration_data.get('brier_score_raw', 'N/A')
        brier_cal = calibration_data.get('brier_score_calibrated', 'N/A')
        ece_raw = calibration_data.get('ece_raw', 'N/A')
        ece_cal = calibration_data.get('ece_calibrated', 'N/A')

        sections.append(f"""
        <div class="section">
            <h3>🎯 Calibration Metrics</h3>
            <table>
                <tr><th>Metric</th><th>Raw</th><th>Calibrated</th></tr>
                <tr><td>Brier Score</td><td>{brier_raw}</td><td>{brier_cal}</td></tr>
                <tr><td>ECE</td><td>{ece_raw}</td><td>{ece_cal}</td></tr>
            </table>
        </div>
        """)

    return ''.join(sections)


def generate_html_report(
    run_id: int,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
    holdout_metric: Optional[float] = None,
    holdout_equity: Optional[float] = None,
    calibration_data: Optional[Dict[str, Any]] = None,
    feature_importance: Optional[Dict[str, Any]] = None,
    feature_usage: Optional[Dict[str, Any]] = None,
    monthly_returns: Optional[List[float]] = None,
    yearly_returns: Optional[List[float]] = None,
    benchmark_return: Optional[float] = None,
    created_at: Optional[str] = None,
    notes: Optional[List[str]] = None,
) -> str:
    """
    生成 HTML 报告。

    Args:
        run_id: 运行 ID
        config: 配置字典
        metrics: 风险指标字典
        equity_curve: 权益曲线 DataFrame
        trades: 交易记录 DataFrame
        initial_cash: 初始资金
        holdout_metric: Holdout 集指标
        holdout_equity: Holdout 集权益
        calibration_data: 校准数据
        feature_importance: 特征重要性数据
        feature_usage: 特征使用统计
        monthly_returns: 月度收益列表
        yearly_returns: 年度收益列表
        benchmark_return: 基准收益
        created_at: 创建时间
        notes: 备注列表

    Returns:
        HTML 字符串
    """
    if created_at is None:
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    plots = {}

    plots['equity'] = _create_equity_curve_plot(equity_curve, initial_cash)
    plots['drawdown'] = _create_drawdown_plot(equity_curve)
    plots['trade_dist'] = _create_trade_distribution_plot(trades)
    plots['feature_importance'] = _create_feature_importance_plot(feature_importance) if feature_importance else None
    plots['calibration'] = _create_calibration_plot(calibration_data) if calibration_data else None

    trade_count = len(trades) if trades is not None and not trades.empty else 0
    win_rate = metrics.get('win_rate', 0) * 100 if metrics.get('win_rate') else 0
    total_return = metrics.get('total_return_pct', 0)
    sharpe = metrics.get('sharpe', 0)
    max_dd = metrics.get('max_drawdown_pct', 0)

    holdout_section = ""
    if holdout_metric is not None:
        holdout_section = f"""
        <div class="alert alert-highlight">
            <h2>🎯 Final Holdout Result</h2>
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-value">{holdout_metric:.4f}</div>
                    <div class="metric-label">Holdout Metric</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{_format_metric(holdout_equity, 'currency')}</div>
                    <div class="metric-label">Holdout Equity</div>
                </div>
            </div>
        </div>
        """

    config_section = _build_config_section(config)

    metrics_section = _build_metrics_section(metrics, benchmark_return, trade_count, win_rate)

    plots_section = _build_plots_section(plots)

    tables_section = _build_tables_section(
        trades, monthly_returns, yearly_returns,
        feature_importance, feature_usage, calibration_data
    )

    notes_section = ""
    if notes:
        notes_section = f"""
        <div class="section">
            <h3>📝 Notes</h3>
            <ul>
                {"".join([f"<li>{note}</li>" for note in notes[-10:]])}
            </ul>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Run #{run_id:03d} - Backtest Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; color: #333; background: #f5f7fa;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 30px; border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .meta {{ opacity: 0.9; font-size: 0.9em; }}

        .alert {{
            padding: 20px; border-radius: 8px; margin-bottom: 20px;
        }}
        .alert-highlight {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }}
        .alert-highlight h2 {{ margin-bottom: 15px; }}

        .metric-row {{
            display: flex; gap: 20px; flex-wrap: wrap;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.2); padding: 20px; border-radius: 8px;
            text-align: center; min-width: 150px;
        }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; }}

        .section {{
            background: white; padding: 25px; border-radius: 10px;
            margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h3 {{
            color: #667eea; margin-bottom: 20px; padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}

        .config-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .config-item {{
            background: #f8f9fa; padding: 12px; border-radius: 6px;
        }}
        .config-item .label {{ color: #666; font-size: 0.85em; }}
        .config-item .value {{ font-weight: 600; color: #333; }}

        .metrics-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }}
        .metric-box {{
            background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;
        }}
        .metric-box .value {{
            font-size: 1.8em; font-weight: bold; color: #667eea;
        }}
        .metric-box .label {{ color: #666; font-size: 0.9em; margin-top: 5px; }}

        .plot-container {{
            text-align: center; margin: 20px 0;
        }}
        .plot-container img {{ max-width: 100%; border-radius: 8px; }}

        table {{
            width: 100%; border-collapse: collapse; margin: 15px 0;
        }}
        th, td {{
            padding: 12px; text-align: left; border-bottom: 1px solid #ddd;
        }}
        th {{ background: #667eea; color: white; }}
        tr:hover {{ background: #f8f9fa; }}

        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}

        .two-column {{
            display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
        }}
        @media (max-width: 768px) {{
            .two-column {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Run #{run_id:03d} - Backtest Report</h1>
            <div class="meta">Generated: {created_at}</div>
        </div>

        {holdout_section}

        <div class="section">
            <h3>📈 Key Performance Metrics</h3>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="value {'positive' if total_return > 0 else 'negative'}">{_format_metric(total_return, 'percent')}</div>
                    <div class="label">Total Return</div>
                </div>
                <div class="metric-box">
                    <div class="value">{_format_metric(sharpe)}</div>
                    <div class="label">Sharpe Ratio</div>
                </div>
                <div class="metric-box">
                    <div class="value {'negative'}">{_format_metric(max_dd, 'percent')}</div>
                    <div class="label">Max Drawdown</div>
                </div>
                <div class="metric-box">
                    <div class="value">{trade_count}</div>
                    <div class="label">Total Trades</div>
                </div>
                <div class="metric-box">
                    <div class="value {'positive' if win_rate > 50 else ''}">{win_rate:.1f}%</div>
                    <div class="label">Win Rate</div>
                </div>
                <div class="metric-box">
                    <div class="value">{_format_metric(metrics.get('annual_return_pct', 0), 'percent')}</div>
                    <div class="label">Annual Return</div>
                </div>
                <div class="metric-box">
                    <div class="value">{_format_metric(metrics.get('annual_vol_pct', 0), 'percent')}</div>
                    <div class="label">Annual Volatility</div>
                </div>
                <div class="metric-box">
                    <div class="value">{_format_metric(metrics.get('calmar', 0))}</div>
                    <div class="label">Calmar Ratio</div>
                </div>
            </div>
        </div>

        {plots_section}

        <div class="section">
            <h3>⚙️ Configuration Summary</h3>
            {config_section}
        </div>

        {tables_section}

        {notes_section}

    </div>
</body>
</html>"""

    return html


def save_html_report(
    output_dir: str,
    run_id: int,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
    **kwargs
) -> str:
    """
    保存 HTML 报告到文件。

    Args:
        output_dir: 输出目录
        run_id: 运行 ID
        config: 配置字典
        metrics: 风险指标字典
        equity_curve: 权益曲线 DataFrame
        trades: 交易记录 DataFrame
        initial_cash: 初始资金
        **kwargs: 其他参数传递给 generate_html_report

    Returns:
        保存的 HTML 文件路径
    """
    html = generate_html_report(
        run_id=run_id,
        config=config,
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
        initial_cash=initial_cash,
        **kwargs
    )

    html_path = os.path.join(output_dir, f"run_{run_id:03d}_report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return html_path


def generate_leaderboard_html(
    runs: List[Dict[str, Any]],
    title: str = "Experiment Leaderboard"
) -> str:
    """
    生成多 run 对比的 Leaderboard HTML。

    Args:
        runs: 运行结果列表
        title: 页面标题

    Returns:
        HTML 字符串
    """
    rows = []
    for i, run in enumerate(runs):
        rows.append(f"""
        <tr>
            <td>{i+1}</td>
            <td>Run #{run.get('run_id', 'N/A'):03d}</td>
            <td class="positive">{run.get('total_return_pct', 0):+.2f}%</td>
            <td>{run.get('sharpe', 0):.3f}</td>
            <td class="negative">{run.get('max_drawdown_pct', 0):.2f}%</td>
            <td>{run.get('trade_count', 0)}</td>
            <td>{run.get('win_rate', 0)*100:.1f}%</td>
            <td>{run.get('created_at', 'N/A')}</td>
        </tr>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #f5f7fa; }}
        h1 {{ color: #667eea; }}
        table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #667eea; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <h1>🏆 {title}</h1>
    <table>
        <tr>
            <th>Rank</th>
            <th>Run ID</th>
            <th>Total Return</th>
            <th>Sharpe</th>
            <th>Max DD</th>
            <th>Trades</th>
            <th>Win Rate</th>
            <th>Date</th>
        </tr>
        {''.join(rows)}
    </table>
</body>
</html>"""
    return html


def save_leaderboard_html(output_dir: str, runs: List[Dict[str, Any]]) -> str:
    """
    保存 Leaderboard HTML。

    Args:
        output_dir: 输出目录
        runs: 运行结果列表

    Returns:
        保存的文件路径
    """
    html = generate_leaderboard_html(runs)
    path = os.path.join(output_dir, "leaderboard.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


def generate_index_html(
    output_dir: str,
    experiments: List[Dict[str, Any]],
) -> str:
    """
    生成研究归档索引页。

    Args:
        output_dir: 输出目录
        experiments: 实验列表

    Returns:
        生成的索引页路径
    """
    cards = []
    for exp in experiments:
        cards.append(f"""
        <div class="card">
            <h3>Run #{exp.get('run_id', 0):03d}</h3>
            <p>Date: {exp.get('created_at', 'N/A')}</p>
            <p>Return: {exp.get('total_return_pct', 0):+.2f}%</p>
            <p>Sharpe: {exp.get('sharpe', 0):.3f}</p>
            <a href="run_{exp.get('run_id', 0):03d}_report.html">View Report</a>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Archive</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #f5f7fa; }}
        h1 {{ color: #667eea; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h3 {{ color: #667eea; margin-bottom: 10px; }}
        .card a {{ color: #667eea; text-decoration: none; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>📁 Research Archive</h1>
    <div class="grid">
        {''.join(cards)}
    </div>
</body>
</html>"""

    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return index_path


# =============================================================================
# 回测输出保存模块 (原 reporter.py)
# =============================================================================

def escape_excel_formulas(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Prevent Excel from treating strings as formulas (which can trigger repair prompts),
    by prefixing strings starting with = + - @ with a single quote.

    Args:
        df: DataFrame to process
        inplace: If True, modify in place; if False, return a copy

    Returns:
        Processed DataFrame
    """
    if not inplace:
        df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda v: ("'" + v) if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v
            )
    return df


def _hash_dataframe(df: pd.DataFrame) -> str:
    """
    对 DataFrame 内容做 SHA-256 哈希，用于检测数据变化。
    使用 pandas 内置哈希（比 to_csv 快约 10x），结果为 16 进制字符串。
    """
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


def _next_run_id(output_dir: str) -> int:
    """
    扫描 output_dir 中已有的 run_XXX_config.json 或 exp_XXX 目录，
    返回下一个可用的 run_id（从 1 开始）。
    若目录为空则返回 1。
    """
    os.makedirs(output_dir, exist_ok=True)
    existing = []
    for fn in os.listdir(output_dir):
        if fn.startswith("run_") and fn.endswith("_config.json"):
            try:
                n = int(fn.split("_")[1])
                existing.append(n)
            except Exception:
                pass
        if fn.startswith("exp_") and os.path.isdir(os.path.join(output_dir, fn)):
            try:
                n = int(fn.split("_")[1])
                existing.append(n)
            except Exception:
                pass
    return (max(existing) + 1) if existing else 1


def find_latest_run(output_dir: str) -> Optional[Tuple[int, str, str]]:
    """
    查找最新的运行记录。

    Args:
        output_dir: 输出目录

    Returns:
        (run_id, config_path, xlsx_path) 元组，若未找到则返回 None
    """
    if not os.path.isdir(output_dir):
        return None

    candidates = []
    for fn in os.listdir(output_dir):
        if fn.startswith("run_") and fn.endswith("_config.json"):
            try:
                run_id = int(fn.split("_")[1])
                cfg_path = os.path.join(output_dir, fn)
                xlsx_path = os.path.join(output_dir, f"run_{run_id:03d}.xlsx")
                candidates.append((run_id, cfg_path, xlsx_path))
            except Exception:
                continue

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: x[0])[-1]


def save_run_outputs(
    output_dir: str,
    df_clean: pd.DataFrame,
    log_notes: List[str],
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    config: Dict[str, object],
    feature_meta: Dict[str, object],
    search_log: pd.DataFrame,
    model_params: Optional[Dict[str, object]] = None,
    feature_usage_stats: Optional[Dict[str, Any]] = None,
    best_model_importance: Optional[Dict[str, Any]] = None,
    use_regime_analysis: bool = False,
    regime_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, int]:
    """
    保存回测运行输出到 Excel 和 HTML 报告。

    Args:
        output_dir: 输出目录
        df_clean: 清洗后的数据 DataFrame
        log_notes: 日志备注列表
        trades: 交易记录 DataFrame
        equity_curve: 权益曲线 DataFrame
        config: 配置字典
        feature_meta: 特征元数据字典
        search_log: 搜索日志 DataFrame
        model_params: 模型参数字典
        feature_usage_stats: 特征使用统计
        best_model_importance: 最佳模型特征重要性
        use_regime_analysis: 是否使用 Regime 分析
        regime_config: Regime 分析配置

    Returns:
        (excel_path, config_path, run_id) 元组
    """
    run_id = _next_run_id(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    excel_path = os.path.join(output_dir, f"run_{run_id:03d}.xlsx")
    config_path = os.path.join(output_dir, f"run_{run_id:03d}_config.json")

    df_hash = _hash_dataframe(df_clean)

    # ---------- build config rows FIRST ----------
    config_blob = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_hash": df_hash,
        "best_config": config,
        "feature_meta": feature_meta,
        "notes": {
            "execution_assumption": "Signal uses day t data; order executes on t+1 at t+1 open price (open_qfq). P1-3: changed from t close to t+1 open to remove forward bias.",
            "a_share_constraints": "Long-only, buy before sell, no short, min 100 shares, full-in/out, T+1 approximated via min_hold_days>=1.",
        },
    }

    config_rows = []
    config_rows.append(("run_id", run_id))
    config_rows.append(("created_at", config_blob["created_at"]))
    config_rows.append(("data_hash", df_hash))
    config_rows.append(("split_mode", config.get("split_mode", "")))

    for k, v in config.get("feature_config", {}).items():
        config_rows.append((f"feature.{k}", str(v)))
    for k, v in config.get("model_config", {}).items():
        config_rows.append((f"model.{k}", str(v)))
    for k, v in config.get("trade_config", {}).items():
        config_rows.append((f"trade.{k}", str(v)))
    for k, v in config.get("calibration_config", {}).items():
        config_rows.append((f"calibration.{k}", str(v)))

    config_rows.append(("constraint.min_closed_trades_per_fold", MIN_CLOSED_TRADES_PER_FOLD))
    config_rows.append(("penalty.target_closed_trades_per_fold", TARGET_CLOSED_TRADES_PER_FOLD))
    config_rows.append(("penalty.lambda_trade_penalty", LAMBDA_TRADE_PENALTY))

    config_rows.append(("dpoint_definition", feature_meta.get("dpoint_explainer", "")))
    config_df = pd.DataFrame(config_rows, columns=["key", "value"])

    notes_df = pd.DataFrame({"notes": log_notes})

    # ---------- escape Excel formulas BEFORE writing Excel ----------
    # 使用 inplace=True 减少 DataFrame 复制，优化内存效率
    escape_excel_formulas(trades, inplace=True)
    escape_excel_formulas(equity_curve, inplace=True)
    escape_excel_formulas(config_df, inplace=True)
    escape_excel_formulas(notes_df, inplace=True)
    escape_excel_formulas(search_log, inplace=True)

    model_params_effective = model_params
    if model_params_effective is None and isinstance(feature_meta, dict):
        model_params_effective = feature_meta.get("model_params")

    model_params_df: Optional[pd.DataFrame] = None
    if isinstance(model_params_effective, dict):
        feature_names = list(model_params_effective.get("feature_names", []))
        coef = list(model_params_effective.get("coef", []))
        scaler_mean = model_params_effective.get("mean", model_params_effective.get("scaler_mean", []))
        scaler_scale = model_params_effective.get("scale", model_params_effective.get("scaler_scale", []))
        scaler_mean = list(scaler_mean) if isinstance(scaler_mean, (list, tuple)) else []
        scaler_scale = list(scaler_scale) if isinstance(scaler_scale, (list, tuple)) else []

        n = max(len(feature_names), len(coef), len(scaler_mean), len(scaler_scale))
        rows = []
        for i in range(n):
            rows.append(
                {
                    "feature_name": feature_names[i] if i < len(feature_names) else "",
                    "coef": coef[i] if i < len(coef) else "",
                    "scaler_mean": scaler_mean[i] if i < len(scaler_mean) else "",
                    "scaler_scale": scaler_scale[i] if i < len(scaler_scale) else "",
                }
            )

        intercept = model_params_effective.get("intercept")
        if intercept is not None:
            rows.append(
                {
                    "feature_name": "__intercept__",
                    "coef": intercept,
                    "scaler_mean": "",
                    "scaler_scale": "",
                }
            )

        if rows:
            model_params_df = pd.DataFrame(
                rows,
                columns=["feature_name", "coef", "scaler_mean", "scaler_scale"],
            )
            escape_excel_formulas(model_params_df, inplace=True)

    # ---------- write config json (ONLY json) ----------
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_blob, f, ensure_ascii=False, indent=2)

    # ---------- P0: 计算统一风险指标 ----------
    # 获取初始资金
    initial_cash = float(config.get("trade_config", {}).get("initial_cash", 100000.0))

    # 准备 benchmark 数据
    benchmark_curve = None
    if "bnh_equity" in equity_curve.columns:
        benchmark_curve = pd.DataFrame({"bnh_equity": equity_curve["bnh_equity"]})

    # 计算风险指标
    risk_metrics = calculate_risk_metrics(
        equity_curve=equity_curve,
        trades=trades,
        initial_cash=initial_cash,
        benchmark_curve=benchmark_curve,
    )

    # P0: 计算日收益率并添加到 equity_curve
    equity_with_returns = equity_curve.copy()
    if "total_equity" in equity_with_returns.columns:
        equity_with_returns["daily_return"] = equity_with_returns["total_equity"].pct_change()
        # 补齐第一行
        equity_with_returns.loc[equity_with_returns.index[0], "daily_return"] = 0.0

    # P2: 计算 regime 指标
    regime_metrics = calculate_regime_metrics(equity_curve, trades, initial_cash)

    # P2: 计算交易分布
    trade_dist = calculate_trade_distribution(trades, equity_curve)

    # 格式化风险指标为 DataFrame
    risk_metrics_rows = []
    for key, value in risk_metrics.items():
        if isinstance(value, list):
            continue  # 跳过列表类型
        risk_metrics_rows.append({"metric": key, "value": value})
    risk_metrics_df = pd.DataFrame(risk_metrics_rows)

    # P2: Regime 指标 DataFrame
    regime_rows = []
    for regime, reg_metrics in regime_metrics.items():
        for k, v in reg_metrics.items():
            regime_rows.append({"regime": regime, "metric": k, "value": v})
    regime_df = pd.DataFrame(regime_rows) if regime_rows else None

    # P0: 使用新的 RegimeDetector 进行更详细的分层分析
    regime_analysis_df = None
    regime_visualization_df = None
    if use_regime_analysis and df_clean is not None and not df_clean.empty:
        try:
            detector = RegimeDetector(
                ma_short=regime_config.get("ma_short", 5) if regime_config else 5,
                ma_long=regime_config.get("ma_long", 20) if regime_config else 20,
                vol_window=regime_config.get("vol_window", 20) if regime_config else 20,
                vol_high_threshold=regime_config.get("vol_high_threshold", 0.20) if regime_config else 0.20,
                vol_low_threshold=regime_config.get("vol_low_threshold", 0.10) if regime_config else 0.10,
            )

            if "close" in df_clean.columns:
                regimes = detector.fit_predict(df_clean)

                regime_labels = regimes["combined"]

                detailed_regime_metrics = compute_regime_metrics(
                    equity_curve, trades, initial_cash, regime_labels, "combined"
                )

                regime_analysis_rows = []
                for regime_name, metrics in detailed_regime_metrics.items():
                    regime_analysis_rows.append({
                        "regime": regime_name,
                        "n_days": metrics.get("n_days", 0),
                        "total_return_pct": metrics.get("total_return_pct", 0),
                        "sharpe": metrics.get("sharpe", 0),
                        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                        "trade_count": metrics.get("trade_count", 0),
                    })

                if regime_analysis_rows:
                    regime_analysis_df = pd.DataFrame(regime_analysis_rows)

                vis_df = create_regime_visualization(df_clean, regimes)
                if "regime" in vis_df.columns:
                    regime_visualization_df = vis_df[["price", "ma_5", "ma_20", "volatility", "regime", "regime_color"]].reset_index()
        except Exception as e:
            logger.warning("Regime analysis failed: %s", e)

    # P2: 交易分布 DataFrame
    dist_rows = []
    for cat, cat_metrics in trade_dist.items():
        for k, v in cat_metrics.items():
            dist_rows.append({"category": cat, "metric": k, "value": v})
    trade_dist_df = pd.DataFrame(dist_rows) if dist_rows else None

    # P1: 校准指标 DataFrame - 从 config 中提取校准信息
    calibration_config = config.get("calibration_config", {})
    calibration_rows = []
    if calibration_config:
        for k, v in calibration_config.items():
            calibration_rows.append({"metric": f"calibration.{k}", "value": str(v)})

    # 从 holdout_calibration_comparison 中提取校准对比信息（如果在 feature_meta 中）
    if isinstance(feature_meta, dict):
        holdout_cal = feature_meta.get("holdout_calibration_comparison", {})
        if holdout_cal:
            calibration_rows.append({"metric": "holdout.comparison.available", "value": "true"})
            for k, v in holdout_cal.items():
                if isinstance(v, (int, float)):
                    calibration_rows.append({"metric": f"holdout.{k}", "value": float(v)})
                else:
                    calibration_rows.append({"metric": f"holdout.{k}", "value": str(v)})

    calibration_df = pd.DataFrame(calibration_rows) if calibration_rows else None

    # 格式化摘要
    metrics_summary = format_metrics_summary(risk_metrics)

    # ---------- write Excel ----------
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        trades.to_excel(writer, sheet_name="Trades", index=False)
        equity_with_returns.to_excel(writer, sheet_name="EquityCurve", index=False)  # P0: 包含 daily_return
        config_df.to_excel(writer, sheet_name="Config", index=False)
        notes_df.to_excel(writer, sheet_name="Log", index=False, startrow=0)

        startrow = len(notes_df) + 3
        search_log.to_excel(writer, sheet_name="Log", index=False, startrow=startrow)

        if model_params_df is not None:
            model_params_df.to_excel(writer, sheet_name="ModelParams", index=False)

        # P0: 新增 RiskMetrics sheet
        risk_metrics_df.to_excel(writer, sheet_name="RiskMetrics", index=False)

        # P2: Regime 分析 sheet
        if regime_df is not None and not regime_df.empty:
            regime_df.to_excel(writer, sheet_name="RegimeAnalysis", index=False)

        # P0: 详细 Regime 分层评估 sheet
        if regime_analysis_df is not None and not regime_analysis_df.empty:
            regime_analysis_df.to_excel(writer, sheet_name="RegimeStratified", index=False)

        # P0: Regime 可视化数据 sheet
        if regime_visualization_df is not None and not regime_visualization_df.empty:
            regime_visualization_df.to_excel(writer, sheet_name="RegimeVisualization", index=False)

        # P2: 交易分布 sheet
        if trade_dist_df is not None and not trade_dist_df.empty:
            trade_dist_df.to_excel(writer, sheet_name="TradeDistribution", index=False)

        # P1: 校准指标 sheet
        if calibration_df is not None and not calibration_df.empty:
            calibration_df.to_excel(writer, sheet_name="CalibrationMetrics", index=False)

        # P0: 特征使用频率 sheet
        if feature_usage_stats:
            fus = feature_usage_stats
            fus_rows = []
            fus_rows.append({"stat": "total_candidates", "value": fus.get("total_candidates", 0)})
            group_usage = fus.get("group_usage", {})
            for key, data in group_usage.items():
                fus_rows.append({"stat": key, "value": f"{data.get('frequency', 0)*100:.2f}%", "count": data.get("count", 0)})
            fus_df = pd.DataFrame(fus_rows)
            fus_df.to_excel(writer, sheet_name="FeatureUsage", index=False)

        # P0-P1: 最佳模型特征重要性 sheet
        if best_model_importance:
            bmi = best_model_importance
            bmi_rows = []
            bmi_rows.append({"type": "method", "value": bmi.get("method", "")})

            ranking = bmi.get("ranking", [])
            for item in ranking:
                bmi_rows.append({
                    "type": "feature",
                    "rank": item.get("rank", ""),
                    "name": item.get("feature", ""),
                    "importance": item.get("importance", ""),
                })

            group_ranking = bmi.get("feature_group_ranking", [])
            for item in group_ranking:
                bmi_rows.append({
                    "type": "group",
                    "rank": item.get("rank", ""),
                    "name": item.get("group", ""),
                    "importance": item.get("importance", ""),
                })

            if bmi_rows:
                bmi_df = pd.DataFrame(bmi_rows)
                bmi_df.to_excel(writer, sheet_name="FeatureImportance", index=False)

    html_path = None
    if use_regime_analysis or True:
        try:
            initial_cash = float(config.get("trade_config", {}).get("initial_cash", 100000.0))

            holdout_metric = None
            holdout_equity = None
            if isinstance(feature_meta, dict):
                holdout_metric = feature_meta.get("holdout_metric")
                holdout_equity = feature_meta.get("holdout_equity")

            calibration_data = None
            if isinstance(feature_meta, dict):
                calibration_data = feature_meta.get("holdout_calibration_comparison")

            html_path = save_html_report(
                output_dir=output_dir,
                run_id=run_id,
                config=config,
                metrics=risk_metrics,
                equity_curve=equity_curve,
                trades=trades,
                initial_cash=initial_cash,
                holdout_metric=holdout_metric,
                holdout_equity=holdout_equity,
                calibration_data=calibration_data,
                feature_importance=best_model_importance,
                feature_usage=feature_usage_stats,
                monthly_returns=risk_metrics.get("monthly_returns"),
                yearly_returns=risk_metrics.get("yearly_returns"),
                benchmark_return=risk_metrics.get("bnh_return"),
                created_at=config_blob.get("created_at"),
                notes=log_notes if isinstance(log_notes, list) else None,
            )
            logger.info("REPORT HTML report saved: %s", html_path)
        except Exception as e:
            logger.warning("REPORT Failed to generate HTML report: %s", e)

    return excel_path, config_path, run_id


def generate_multi_run_report(output_dir: str) -> str:
    """
    P2: 生成多 run 对比报告和索引页。

    Args:
        output_dir: 输出目录

    Returns:
        生成的 leaderboard.html 路径
    """
    runs = []

    for fn in os.listdir(output_dir):
        if fn.startswith("run_") and fn.endswith("_config.json"):
            try:
                config_path = os.path.join(output_dir, fn)
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_blob = json.load(f)

                run_id = config_blob.get("run_id")
                run_dir = output_dir

                excel_path = os.path.join(output_dir, f"run_{run_id:03d}.xlsx")
                if not os.path.exists(excel_path):
                    continue

                from evaluation import calculate_risk_metrics

                try:
                    equity_df = pd.read_excel(excel_path, sheet_name="EquityCurve")
                    trades_df = pd.read_excel(excel_path, sheet_name="Trades")

                    initial_cash = config_blob.get("best_config", {}).get("trade_config", {}).get("initial_cash", 100000.0)

                    risk_metrics = calculate_risk_metrics(
                        equity_curve=equity_df,
                        trades=trades_df,
                        initial_cash=initial_cash,
                    )

                    runs.append({
                        "run_id": run_id,
                        "created_at": config_blob.get("created_at", ""),
                        "total_return_pct": risk_metrics.get("total_return_pct", 0),
                        "sharpe": risk_metrics.get("sharpe", 0),
                        "max_drawdown_pct": risk_metrics.get("max_drawdown_pct", 0),
                        "trade_count": risk_metrics.get("trade_count", 0),
                        "win_rate": risk_metrics.get("win_rate", 0),
                        "annual_return_pct": risk_metrics.get("annual_return_pct", 0),
                    })
                except Exception as e:
                    logger.warning("Failed to load run %s: %s", run_id, e)
            except Exception as e:
                logger.warning("Failed to load config %s: %s", fn, e)

    if not runs:
        logger.info("No runs found for multi-run report")
        return ""

    runs_sorted = sorted(runs, key=lambda x: x.get('total_return_pct', 0), reverse=True)

    leaderboard_path = save_leaderboard_html(output_dir, runs_sorted)
    logger.info("REPORT Leaderboard saved: %s", leaderboard_path)

    return leaderboard_path


# =============================================================================
# 公开 API 导出列表
# =============================================================================

__all__ = [
    # 主要输出函数
    'save_run_outputs',
    'generate_multi_run_report',
    
    # HTML 报告生成
    'generate_html_report',
    'save_html_report',
    
    # Leaderboard 和索引
    'generate_leaderboard_html',
    'save_leaderboard_html',
    'generate_index_html',
    
    # 工具函数
    'escape_excel_formulas',
    'find_latest_run',
]

# html_reporter.py
"""
HTML 报告生成模块。

P0:
    - 生成 HTML summary 报告
    - 包含 run summary, config summary, key metrics
    - equity curve, drawdown curve, trade summary
    - final holdout 结果放在最显眼位置

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
"""
from __future__ import annotations

import base64
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _save_plot_to_base64(fig, format='png', dpi=100):
    """将 matplotlib 图表保存为 base64 编码。"""
    import io
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
    """保存 HTML 报告到文件。"""
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
    """生成多 run 对比的 Leaderboard HTML。"""
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
    """保存 Leaderboard HTML。"""
    html = generate_leaderboard_html(runs)
    path = os.path.join(output_dir, "leaderboard.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


def generate_index_html(
    output_dir: str,
    experiments: List[Dict[str, Any]],
) -> str:
    """生成研究归档索引页。"""
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

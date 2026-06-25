"""
generate_report.py — 生成每日预测报告 + 过去7天回测准确率
输出: data/daily_report.md
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime

from src.data_fetcher import fetch_us_stocks, fetch_korea_stocks, fetch_ashare_stocks
from src.processor import (
    align_us_to_ashare, build_sector_features,
    build_ashare_baskets, build_full_dataset,
)
from src.model import train_sector_models
from src.stocks_universe import SECTORS

# ── 配置 ──────────────────────────────────────────────────────────
BACKTEST_DAYS = 7        # 回测最近N个A股交易日
REPORT_FILE   = "data/daily_report.md"
TODAY_STR     = datetime.now().strftime("%Y-%m-%d")


def signal_emoji(prob: float) -> str:
    if prob >= 0.60: return "🟢 强多"
    if prob >= 0.55: return "🟩 偏多"
    if prob <= 0.40: return "🔴 强空"
    if prob <= 0.45: return "🟧 偏空"
    return "⬜ 中性"

def bar(prob: float, width: int = 10) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def build_backtest_table(df: pd.DataFrame, sector_results: dict) -> dict:
    """
    对过去 BACKTEST_DAYS 个交易日，用已训练好的模型做预测，与实际对比
    返回 {sector_key: DataFrame with columns [date, prob_up, actual_label, correct]}
    """
    from src.processor import get_feature_columns
    feature_cols = get_feature_columns(df)
    backtest = {}

    for sk, res in sector_results.items():
        label_col = f"{sk}_label"
        if label_col not in df.columns:
            continue

        subset = df[feature_cols + [label_col]].dropna()
        if len(subset) < BACKTEST_DAYS + 10:
            continue

        # 取最后 BACKTEST_DAYS 行作为回测窗口
        bt_window = subset.iloc[-(BACKTEST_DAYS):]
        X_bt = bt_window[feature_cols]
        y_bt = bt_window[label_col].astype(int)

        probs = res["model"].predict_proba(X_bt)[:, 1]
        preds = (probs >= 0.5).astype(int)
        correct = (preds == y_bt.values).astype(int)

        backtest[sk] = pd.DataFrame({
            "date":         bt_window.index.strftime("%Y-%m-%d"),
            "prob_up":      probs,
            "pred_dir":     ["↑" if p == 1 else "↓" for p in preds],
            "actual_label": y_bt.values,
            "actual_dir":   ["↑" if a == 1 else "↓" for a in y_bt.values],
            "correct":      correct,
        })

    return backtest


def generate_markdown(sector_results: dict, backtest: dict, df: pd.DataFrame) -> str:
    lines = []

    lines += [
        f"# GPU/AI 产业链 · 早盘预测报告",
        f"",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} | 基于 Lead-Lag 跨市场效应（美/韩股 → A股）",
        f"",
        "---",
        "",
    ]

    # ── Part 1：今日预测 ──────────────────────────────────────────
    lines += [
        "## 📊 今日预测",
        f"",
        f"**预测日期：{TODAY_STR}**（美股 T-1 信号 + 韩股 T 日跳空信号）",
        "",
        "| 板块 | 涵盖环节 | 上涨概率 | 信号 | 强度 |",
        "|------|---------|:--------:|------|------|",
    ]

    for sk, res in sector_results.items():
        prob = res["prob_up"]
        lines.append(
            f"| {res['sector_name']} | {res['sector_desc'][:28]}… | "
            f"**{prob:.1%}** | {signal_emoji(prob)} | `{bar(prob)}` |"
        )

    lines += ["", "### 个股预测明细", ""]

    for sk, res in sector_results.items():
        lines.append(f"**{res['sector_name']}**")
        lines.append("")
        lines.append("| 代码 | 名称 | 上涨概率 | 信号 |")
        lines.append("|------|------|:--------:|------|")

        for code, name in SECTORS[sk]["a"].items():
            # 所有个股使用板块概率（板块级预测）
            prob = res["prob_up"]
            lines.append(f"| {code} | {name} | {prob:.1%} | {signal_emoji(prob)} |")
        lines.append("")

    # 关键驱动因子
    lines += ["### 🔑 关键驱动因子（各板块 Top 3）", ""]
    lines += ["| 板块 | 因子1 | 因子2 | 因子3 |", "|------|-------|-------|-------|"]
    for sk, res in sector_results.items():
        top3 = [f"`{f}` {v:.1%}" for f, v in res["top_features"][:3]]
        while len(top3) < 3:
            top3.append("—")
        lines.append(f"| {res['sector_name']} | {top3[0]} | {top3[1]} | {top3[2]} |")

    lines += ["", "---", ""]

    # ── Part 2：过去7天回测准确率 ─────────────────────────────────
    lines += [
        "## 📈 过去 7 个交易日回测准确率",
        "",
        f"> 使用已训练模型，对最近 {BACKTEST_DAYS} 个A股交易日做回测，验证 Lead-Lag 信号有效性",
        "",
    ]

    # 汇总准确率表
    lines += ["### 板块准确率汇总", ""]
    lines += ["| 板块 | 准确次数 | 总次数 | 近7日准确率 | 整体历史准确率 |",
              "|------|:--------:|:------:|:-----------:|:--------------:|"]

    for sk, bt_df in backtest.items():
        if sk not in sector_results:
            continue
        n_correct = bt_df["correct"].sum()
        n_total   = len(bt_df)
        bt_acc    = n_correct / n_total if n_total > 0 else 0
        hist_acc  = sector_results[sk]["accuracy"]
        lines.append(
            f"| {sector_results[sk]['sector_name']} | {n_correct} | {n_total} | "
            f"**{bt_acc:.0%}** | {hist_acc:.1%} |"
        )

    lines += [""]

    # 逐板块逐日明细
    lines += ["### 逐日明细", ""]

    for sk, bt_df in backtest.items():
        if sk not in sector_results:
            continue
        lines.append(f"**{sector_results[sk]['sector_name']}**")
        lines.append("")
        lines.append("| 日期 | 预测概率 | 预测方向 | 实际方向 | 是否正确 |")
        lines.append("|------|:--------:|:--------:|:--------:|:--------:|")
        for _, row in bt_df.iterrows():
            tick = "✅" if row["correct"] == 1 else "❌"
            lines.append(
                f"| {row['date']} | {row['prob_up']:.1%} | "
                f"{row['pred_dir']} | {row['actual_dir']} | {tick} |"
            )
        lines.append("")

    # ── Part 3：免责声明 ──────────────────────────────────────────
    lines += [
        "---",
        "",
        "## ⚠️ 免责声明",
        "",
        "- 本报告基于统计模型，历史规律不代表未来表现",
        "- 模型仅使用价格信号，未考虑基本面、政策、突发事件等因素",
        "- 准确率低于 60% 时信号可靠性较低，请谨慎参考",
        "- **本报告不构成任何投资建议，据此操作风险自负**",
        "",
    ]

    return "\n".join(lines)


def main():
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("  GPU/AI 产业链报告生成器")
    print("=" * 60)

    print("\n[1/4] 获取美股数据...")
    us_returns = fetch_us_stocks("2022-01-01")

    print("\n[2/4] 获取韩股数据...")
    kr_gaps = fetch_korea_stocks("2022-01-01")

    print("\n[3/4] 获取A股数据...")
    ashare_data = fetch_ashare_stocks("2022-01-01")

    print("\n[4/4] 对齐 & 训练模型...")
    all_dates = None
    for d in ashare_data.values():
        idx = d["return"].dropna().index
        all_dates = idx if all_dates is None else all_dates.union(idx)
    all_dates = all_dates.sort_values()

    us_aligned      = align_us_to_ashare(us_returns, all_dates)
    sector_features = build_sector_features(us_aligned, kr_gaps, all_dates)
    sector_targets, stock_labels = build_ashare_baskets(ashare_data, all_dates)
    df = build_full_dataset(us_aligned, kr_gaps, sector_features, sector_targets, stock_labels)
    df.to_csv("data/full_dataset.csv")

    print("\n训练模型...")
    sector_results = train_sector_models(df)

    print("\n生成回测数据...")
    backtest = build_backtest_table(df, sector_results)

    print("\n生成报告...")
    md_content = generate_markdown(sector_results, backtest, df)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n✅ 报告已生成: {REPORT_FILE}")
    print(f"   板块数: {len(sector_results)}, 回测天数: {BACKTEST_DAYS}")
    
    # 微信推送
    from src.notifier import send_wechat_webhook
    send_wechat_webhook(md_content)


if __name__ == "__main__":
    main()

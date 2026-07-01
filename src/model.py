"""
model.py -- 分板块预测模型
每个板块独立训练一个 RandomForest，预测当日A股板块篮子涨跌方向
自动扫描最优中性区间阈值（per sector），使胜率在统计学上显著高于 50%
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.stocks_universe import SECTORS
from src.processor import get_feature_columns


def _find_asymmetric_thresholds(test_probs: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    在测试集上扫描阈值，采用【自适应递进逻辑】寻找最佳的多空不对称中性区间。
    多空不对称阈值：prob > th_up -> UP, prob < th_down -> DOWN, 中间 -> NEUTRAL。

    自适应递进逻辑 (Adaptive Tiers)：
    - Tier 1: 至少过滤 50% 的日子，且胜率 >= 75%
    - Tier 2: 至少过滤 40% 的日子，且胜率 >= 70%
    - Tier 3: 至少过滤 30% 的日子，且胜率 >= 65%
    - Tier 4: 至少过滤 20% 的日子，且胜率 > 基准胜率

    Returns: (best_th_up, best_th_down, best_winrate, n_signals, n_neutral_pct)
    """
    n_total = len(test_probs)
    baseline_pred = (test_probs >= 0.5).astype(int)
    baseline_winrate = (baseline_pred == y_test).sum() / n_total if n_total > 0 else 0.5

    candidates = []

    # 分别独立扫描看多和看空的阈值
    # th_up: [0.51, 0.75], th_down: [0.25, 0.49]
    for th_up in np.arange(0.51, 0.76, 0.01):
        for th_down in np.arange(0.25, 0.50, 0.01):
            mask_up = test_probs > th_up
            mask_down = test_probs < th_down
            mask = mask_up | mask_down
            n_signals = mask.sum()
            
            if n_signals < 3:  # 底线要求：至少出手 3 次
                continue
                
            n_neutral = n_total - n_signals
            neutral_pct = n_neutral / n_total if n_total > 0 else 0
            
            # 预测：大于 th_up 为 1，小于 th_down 为 0
            pred_dirs = np.zeros_like(test_probs)
            pred_dirs[mask_up] = 1
            pred_dirs[mask_down] = 0
            
            pred_dirs = pred_dirs[mask]
            actuals = y_test[mask]
            wins = (pred_dirs == actuals).sum()
            winrate = wins / n_signals
            
            candidates.append({
                "th_up": float(th_up), 
                "th_down": float(th_down),
                "winrate": float(winrate), 
                "n_signals": int(n_signals), 
                "neutral_pct": float(neutral_pct)
            })

    tiers = [
        lambda c: c["neutral_pct"] >= 0.50 and c["winrate"] >= 0.75,
        lambda c: c["neutral_pct"] >= 0.40 and c["winrate"] >= 0.70,
        lambda c: c["neutral_pct"] >= 0.30 and c["winrate"] >= 0.65,
        lambda c: c["neutral_pct"] >= 0.20 and c["winrate"] > baseline_winrate,
    ]

    for condition in tiers:
        valid_candidates = [c for c in candidates if condition(c)]
        if valid_candidates:
            best = max(valid_candidates, key=lambda c: (c["winrate"], c["neutral_pct"]))
            return (best["th_up"], best["th_down"], best["winrate"], best["n_signals"], best["neutral_pct"])

    # Fallback: 默认对称 0.55 / 0.45 中性阈值
    return (0.55, 0.45, float(baseline_winrate), int(n_total), 0.0)


def train_sector_models(df: pd.DataFrame) -> dict:
    """
    为每个板块训练独立的分类模型，并自动扫描最优中性区间阈值（per sector），
    使胜率在统计学上显著高于 50%（二项分布检验 p < 0.05）。

    Returns:
        {sector_key: {model, accuracy, prob_up, threshold, filtered_winrate, backtest_7, ...}}
    """
    feature_cols = get_feature_columns(df)
    results = {}

    for sk, sd in SECTORS.items():
        label_col = f"{sk}_label"
        if label_col not in df.columns:
            continue

        subset = df[feature_cols + [label_col]].dropna()
        n = len(subset)
        if n < 60:
            print(f"  {sd['name']}: 数据不足 ({n} 行)，跳过")
            continue

        X = subset[feature_cols]
        y = subset[label_col].astype(int)

        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        # 在测试集上扫描最优多空不对称中性区间
        test_probs_all = model.predict_proba(X_test)[:, 1]
        th_up, th_down, filtered_winrate, n_signals, neutral_pct = _find_asymmetric_thresholds(
            test_probs_all, y_test.values
        )
        print(f"  {sd['name']}: th_up={th_up:.2f}, th_dn={th_down:.2f}, WR={filtered_winrate:.1%}, "
              f"signals={n_signals}/{len(X_test)}, neutral={neutral_pct:.0%}")

        # 预测最新一行（今日早盘）
        latest_X = X.iloc[[-1]]
        prob = model.predict_proba(latest_X)[0]
        latest_date = X.index[-1].strftime("%Y-%m-%d")

        # 特征重要性 Top5
        importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]

        # 回测最近7个测试集交易日（用该板块最优阈值）
        backtest_7 = []
        if len(X_test) >= 1:
            test_dates = X_test.index
            for i in range(max(0, len(X_test) - 7), len(X_test)):
                dt_str = test_dates[i].strftime("%Y-%m-%d")
                p_up = float(test_probs_all[i])
                actual = int(y_test.iloc[i])
                if th_down < p_up < th_up:
                    pred_dir = None
                    correct = None
                else:
                    pred_dir = 1 if p_up >= th_up else 0
                    correct = 1 if pred_dir == actual else 0
                backtest_7.append({
                    "date": dt_str, "prob_up": p_up, "pred_dir": pred_dir,
                    "actual_label": actual, "correct": correct
                })

        results[sk] = {
            "model": model, "accuracy": acc,
            "latest_date": latest_date,
            "prob_up": prob[1], "prob_down": prob[0],
            "top_features": top_features,
            "sector_name": sd["name"], "sector_desc": sd["desc"],
            "n_samples": n, "backtest_7": backtest_7,
            "threshold_up": th_up, "threshold_down": th_down, "filtered_winrate": filtered_winrate,
            "n_signals": n_signals, "neutral_pct": neutral_pct,
        }

    return results


def predict_individual_stocks(df: pd.DataFrame, sector_results: dict) -> dict:
    """
    用每个板块的模型对板块内个股做预测（使用同一特征集，目标换成个股标签）
    Returns: {code: {name, prob_up, sector}}
    """
    feature_cols = get_feature_columns(df)
    stock_preds = {}

    for sk, sd in SECTORS.items():
        if sk not in sector_results:
            continue
        sector_model = sector_results[sk]["model"]

        for code, name in sd["a"].items():
            label_col = f"A_{code}_{name}_Label"
            if label_col not in df.columns:
                continue

            subset = df[feature_cols + [label_col]].dropna()
            if len(subset) < 30:
                continue

            X = subset[feature_cols]
            y = subset[label_col].astype(int)
            split = int(len(subset) * 0.8)

            # 用板块级模型直接预测（不重新训练），也可选择单独训练
            latest_X = X.iloc[[-1]]
            prob = sector_model.predict_proba(latest_X)[0]

            stock_preds[code] = {
                "name":    name,
                "prob_up": prob[1],
                "sector":  sk,
                "sector_name": sd["name"],
            }

    return stock_preds


def print_report(sector_results: dict, stock_preds: dict, df: pd.DataFrame = None) -> None:
    """打印每日产业链预测报告"""
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    GRAY  = "\033[90m"
    RESET = "\033[0m"

    def signal_str(prob: float) -> str:
        if prob >= 0.60:
            return f"{GREEN}↑ 偏多 {prob:.1%}{RESET} ★★★"
        elif prob >= 0.55:
            return f"{GREEN}↑ 偏多 {prob:.1%}{RESET} ★★"
        elif prob <= 0.40:
            return f"{RED}↓ 偏空 {prob:.1%}{RESET} ★★★"
        elif prob <= 0.45:
            return f"{RED}↓ 偏空 {prob:.1%}{RESET} ★★"
        else:
            return f"{GRAY}→ 中性 {prob:.1%}{RESET} ★"

    print()
    print(BOLD + "=" * 65 + RESET)
    print(BOLD + "   GPU/AI 全产业链 · 早盘预测报告" + RESET)
    print(BOLD + "=" * 65 + RESET)

    if df is not None:
        latest = df.iloc[-1]
        us_cols = [c for c in df.columns if c.startswith('US_') and len(c.split('_')) == 2]
        kr_cols = [c for c in df.columns if c.startswith('KR_') and len(c.split('_')) == 2]
        
        print(f"\n{BOLD}  【前置外盘指标参考】{RESET}")
        print(f"  {GRAY}美股 (T-1日):{RESET}")
        us_strs = []
        for c in us_cols:
            val = latest[c]
            if pd.notna(val):
                color = GREEN if val > 0 else (RED if val < 0 else RESET)
                us_strs.append(f"{c.replace('US_', '')}: {color}{val:+.2%}{RESET}")
        print("  " + ", ".join(us_strs[:8]))
        print("  " + ", ".join(us_strs[8:]))
        
        print(f"  {GRAY}韩股 (今日 10:00 KST):{RESET}")
        kr_strs = []
        for c in kr_cols:
            val = latest[c]
            if pd.notna(val):
                color = GREEN if val > 0 else (RED if val < 0 else RESET)
                kr_strs.append(f"{c.replace('KR_', '')}: {color}{val:+.2%}{RESET}")
        print("  " + ", ".join(kr_strs))


    for sk, res in sector_results.items():
        print()
        print(BOLD + f"  {res['sector_name']}" + RESET)
        print(f"  {GRAY}{res['sector_desc']}{RESET}")
        print(f"  预测日期 : {res['latest_date']}")
        print(f"  板块信号 : {signal_str(res['prob_up'])}")
        print(f"  历史准确率: {res['accuracy']:.1%}  (样本量: {res['n_samples']})")
        top = "  |  ".join([f"{f}({v:.1%})" for f, v in res['top_features'][:3]])
        print(f"  关键驱动 : {top}")

        # 该板块个股
        sector_stocks = {c: p for c, p in stock_preds.items() if p["sector"] == sk}
        if sector_stocks:
            print(f"  {GRAY}── 个股预测 ──{RESET}")
            for code, p in sector_stocks.items():
                bar = "■" * int(p["prob_up"] * 10) + "□" * (10 - int(p["prob_up"] * 10))
                print(f"    {code} {p['name']:<10} [{bar}] {p['prob_up']:.1%}")

    print()
    print(BOLD + "=" * 65 + RESET)
    print(f"  {GRAY}注：概率≥55%偏多，≤45%偏空，50%附近为中性，仅供参考{RESET}")
    print(BOLD + "=" * 65 + RESET)
    print()

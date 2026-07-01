"""
model.py -- 分板块预测模型
使用 3分类 RandomForestClassifier (偏多 / 偏空 / 中性)
【优化1】实施严格的特征隔离（Feature Isolation），各板块模型只看自己对应上游特征，杜绝全局因子信息泄露。
【优化2】自定义 class_weight={1: 5.0, -1: 5.0, 0: 1.0}，重拳惩罚无脑预测中性的行为。
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.stocks_universe import SECTORS

def get_3class_label(ret: float) -> int:
    """连续收益率转为3分类: 1(偏多), -1(偏空), 0(中性)"""
    if pd.isna(ret):
        return np.nan
    if ret > 0.01:
        return 1
    elif ret < -0.01:
        return -1
    else:
        return 0

def train_sector_models(df: pd.DataFrame) -> dict:
    results = {}

    for sk, sd in SECTORS.items():
        label_col = f"{sk}_label"
        if label_col not in df.columns:
            continue

        # --- 特征隔离 (Feature Isolation) ---
        sector_us_cols = [f"US_{t}" for t in sd["us"].keys()]
        sector_kr_cols = [f"KR_{t.split('.')[0]}" for t in sd["kr"].keys()]
        agg_cols = [f"{sk}_US", f"{sk}_KR"]
        
        my_cols = sector_us_cols + sector_kr_cols + agg_cols
        sector_feature_cols = [c for c in my_cols if c in df.columns]

        subset = df[sector_feature_cols + [label_col]].dropna()
        n = len(subset)
        if n < 60:
            print(f"  {sd['name']}: 数据不足 ({n} 行)，跳过")
            continue

        X = subset[sector_feature_cols]
        y = subset[label_col].apply(get_3class_label)

        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # --- 正则/惩罚机制 ---
        # 如果真是1/-1，模型却保守地报了0，就要吃5倍的Loss惩罚！
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            class_weight={1: 5.0, -1: 5.0, 0: 1.0},
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # 预测最新一行
        latest_X = df[sector_feature_cols].iloc[[-1]]
        pred_class = int(model.predict(latest_X)[0])
        latest_date = df.index[-1].strftime("%Y-%m-%d")

        # 特征重要性 Top3
        importance = dict(zip(sector_feature_cols, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:3]

        results[sk] = {
            "model": model, 
            "accuracy": acc,
            "latest_date": latest_date,
            "pred_class": pred_class,
            "top_features": top_features,
            "sector_name": sd["name"], "sector_desc": sd["desc"],
            "n_samples": n,
        }

    return results

def print_report(sector_results: dict, df: pd.DataFrame = None) -> None:
    """打印每日产业链预测报告"""
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    GRAY  = "\033[90m"
    RESET = "\033[0m"

    def signal_str(pred_class: int) -> str:
        if pred_class == 1:
            return f"{GREEN}↑ 偏多 (预计涨幅 > 1%){RESET} ★★★"
        elif pred_class == -1:
            return f"{RED}↓ 偏空 (预计跌幅 < -1%){RESET} ★★★"
        else:
            return f"{GRAY}→ 中性 (震荡区间 ±1%){RESET} ★"

    print()
    print(BOLD + "=" * 65 + RESET)
    print(BOLD + "   GPU/AI 全产业链 · 早盘预测报告 (特征隔离 + 重力惩罚版)" + RESET)
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
        print(f"  板块信号 : {signal_str(res['pred_class'])}")
        print(f"  测试集准确率: {res['accuracy']:.1%}  (样本量: {res['n_samples']})")
        top = "  |  ".join([f"{f.replace('US_','').replace('KR_','')}({v:.1%})" for f, v in res['top_features']])
        print(f"  专属驱动 : {top}")

    print()
    print(BOLD + "=" * 65 + RESET)
    print(f"  {GRAY}注：各板块已物理隔离无关因子；错判趋势惩罚系数 5.0x{RESET}")
    print(BOLD + "=" * 65 + RESET)
    print()

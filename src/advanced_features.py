import pandas as pd
import numpy as np
from src.feature_map import INDUSTRY_HASHMAP
from src.stocks_universe import SECTORS

def parse_tam(tam_str: str) -> float:
    if not tam_str:
        return 0.0
    return float(tam_str.replace("B", "").replace("+", ""))

# 映射现有个股到 HashMap 的名称，以便提取 TAM 和属性
TICKER_TO_NAME = {
    "SNPS": "新思科技", "CDNS": "楷登电子", "NVDA": "英伟达", "AMD": "AMD",
    "AAPL": "苹果", "QCOM": "高通", "TSM": "台积电(TSMC)", "ASML": "ASML",
    "AMAT": "应用材料", "LRCX": "泛林半导体", "KLAC": "科磊", "MU": "美光",
    "AMKR": "日月光", "SMCI": "超微", "VRT": "维谛(Vertiv)", "COHR": "高意(Coherent)",
    "MRVL": "Marvell", "APH": "安费诺", "LITE": "Lumentum",
    "005930.KS": "三星", "000660.KS": "SK海力士", "009150.KS": "三星电机",
    "688256": "寒武纪", "688047": "海光信息", "688981": "中芯国际", "002371": "北方华创",
    "688012": "中微公司", "688126": "沪硅产业", "600584": "长电科技", "002156": "通富微电",
    "002185": "华天科技", "002436": "兴森科技", "688183": "深南电路", "000977": "浪潮信息",
    "603019": "中科曙光", "300308": "中际旭创", "300394": "天孚", "300502": "新易盛",
    "002179": "中航光电", "002837": "英维克", "601869": "长飞光纤", "600487": "亨通光电",
    "600522": "中天科技"
}

def add_advanced_features(all_data: pd.DataFrame, us_aligned: pd.DataFrame, kr_gaps: pd.DataFrame) -> pd.DataFrame:
    df = all_data.copy()
    
    # 1. 基础类别编码 & TAM加权特征
    # 计算按TAM加权的板块特征
    tam_weights = {}
    for ticker, name in TICKER_TO_NAME.items():
        if name in INDUSTRY_HASHMAP:
            tam_weights[ticker] = parse_tam(INDUSTRY_HASHMAP[name]["tam_2026"])
        else:
            tam_weights[ticker] = 10.0 # 默认权重

    for sk, sd in SECTORS.items():
        # US
        us_cols = [f"US_{t}" for t in sd["us"] if f"US_{t}" in us_aligned.columns]
        if us_cols:
            weights = np.array([tam_weights.get(c.replace("US_", ""), 10.0) for c in us_cols])
            weights = weights / (weights.sum() + 1e-9)
            df[f"ADV_{sk}_US_TAM_Weighted"] = (us_aligned[us_cols] * weights).sum(axis=1)

        # KR
        kr_cols = [f"KR_{t.split('.')[0]}" for t in sd["kr"] if f"KR_{t.split('.')[0]}" in kr_gaps.columns]
        if kr_cols:
            kr_t = [t for t in sd["kr"] if f"KR_{t.split('.')[0]}" in kr_gaps.columns]
            weights = np.array([tam_weights.get(t, 10.0) for t in kr_t])
            weights = weights / (weights.sum() + 1e-9)
            df[f"ADV_{sk}_KR_TAM_Weighted"] = (kr_gaps[kr_cols] * weights).sum(axis=1)

    # 2. 跨市场信号传导 / 国产替代套利特征 (Lead-Lag Momentum)
    # 对于每个板块，计算海外龙头(美/韩) T-1 的动量 (已对齐在 us_aligned, kr_gaps)
    # 取3日滑动均值作为动量
    for sk in SECTORS.keys():
        if f"{sk}_US" in df.columns:
            df[f"ADV_{sk}_US_Momentum_3D"] = df[f"{sk}_US"].rolling(3, min_periods=1).mean()
        if f"{sk}_KR" in df.columns:
            df[f"ADV_{sk}_KR_Momentum_3D"] = df[f"{sk}_KR"].rolling(3, min_periods=1).mean()
            
    # 3. 产业链拓扑与图节点特征 (Upstream Shock)
    # 逻辑: s1(设计) -> s2(制造) -> s3(封装) -> s4(整机) -> s5(连接)
    # 下游特征可以使用上游的信号
    if "s1_design_US" in df.columns:
        df["ADV_s2_mfg_Upstream_Signal"] = df["s1_design_US"].shift(1).fillna(0)
    if "s2_mfg_US" in df.columns:
        df["ADV_s3_pkg_Upstream_Signal"] = df["s2_mfg_US"].shift(1).fillna(0)
    if "s3_pkg_US" in df.columns:
        df["ADV_s4_server_Upstream_Signal"] = df["s3_pkg_US"].shift(1).fillna(0)
    if "s4_server_US" in df.columns:
        df["ADV_s5_connect_Upstream_Signal"] = df["s4_server_US"].shift(1).fillna(0)

    # 4. 卖方研报逻辑特征化 (Rule-based Alpha Flags)
    # 由于这些是静态板块特征，我们可以赋予特定板块Flag，为了作为时序特征，可以与市场波动率或动量结合
    # 这里直接将其作为标量特征
    df["ADV_Flag_Heavy_Moat_s4"] = 1 # AI整机柜
    df["ADV_Flag_Heavy_Moat_s5"] = 1 # 光模块
    df["ADV_Flag_Hidden_Gem_s3"] = 1 # 封装材料
    df["ADV_Flag_Edge_Compute_s1"] = 1 # 端侧NPU
    
    return df

"""
from __future__ import annotations

GPU/AI 算力产业链股票池
基于《全球GPU与AI算力产业链全景图·15大环节规模标定版》
5大板块 · 15大环节
"""

SECTORS = {
    "s1_design": {
        "name": "板块1·设计生态",
        "desc": "EDA / GPU / CPU / NPU 芯片设计（环节1-3）",
        "us": {
            "NVDA":  "英伟达（GPU/AI计算领导者）",
            "AMD":   "AMD（GPU/CPU竞争者）",
            "CDNS":  "Cadence（EDA工具）",
            "SNPS":  "Synopsys（EDA工具）",
        },
        "kr": {},
        "a": {
            "688256": "寒武纪（NPU/AI芯片）",
            "300474": "景嘉微（国产GPU）",
            "688047": "龙芯中科（国产CPU）",
        },
    },

    "s2_mfg": {
        "name": "板块2·制造生态",
        "desc": "晶圆代工 / 光刻设备 / HBM内存（环节4-6）",
        "us": {
            "ASML":  "ASML（光刻机垄断者）",
            "AMAT":  "应用材料（CVD/PVD等设备）",
            "LRCX":  "泛林半导体（刻蚀设备）",
            "KLAC":  "KLA（量测检测设备）",
        },
        "kr": {
            "005930.KS": "三星电子（HBM/代工/内存）",
            "000660.KS": "SK Hynix（HBM No.1）",
        },
        "a": {
            "688981": "中芯国际（国内晶圆代工龙头）",
            "002371": "北方华创（半导体设备龙头）",
            "688012": "中微公司（刻蚀设备）",
            "688126": "沪硅产业（大硅片）",
        },
    },

    "s3_pkg": {
        "name": "板块3·封装互联",
        "desc": "CoWoS先进封装 / ABF基板 / MSAP高端PCB（环节7-9）",
        "us": {
            "AMKR":  "Amkor（先进封装服务商）",
        },
        "kr": {
            "009150.KS": "三星电机（封装基板/MLCC）",
        },
        "a": {
            "600584": "长电科技（先进封装龙头）",
            "002156": "通富微电（先进封装）",
            "002185": "华天科技（封装）",
            "002436": "兴森科技（ABF基板）",
            "688183": "生益电子（高端PCB）",
        },
    },

    "s4_server": {
        "name": "板块4·整机与部署",
        "desc": "AI服务器 ODM整机 / 数据中心基础设施（环节11）",
        "us": {
            "SMCI":  "超微电脑（AI服务器龙头）",
            "VRT":   "Vertiv（数据中心电源/冷却）",
            "HPE":   "HPE（服务器巨头）",
        },
        "kr": {},
        "a": {
            "000977": "浪潮信息（国内服务器龙头）",
            "603019": "中科曙光（高性能服务器）",
        },
    },

    "s5_connect": {
        "name": "板块5·连接与散热",
        "desc": "光模块 / 激光芯片 / 连接器 / 液冷散热 / 光纤光缆（环节12-15，含新增光缆）",
        "us": {
            "COHR":  "Coherent（光模块全球No.1）",
            "MRVL":  "Marvell（DSP/交换芯片）",
            "APH":   "Amphenol（连接器巨头）",
            "LITE":  "Lumentum（EML激光器）",
        },
        "kr": {},
        "a": {
            # 光模块
            "300308": "中际旭创（光模块龙头）",
            "300394": "天孚通信（光器件/CPO）",
            "300502": "新易盛（光模块）",
            # 连接器
            "002179": "中航光电（连接器）",
            # 散热/液冷
            "002837": "英维克（精密空调/液冷）",
            # 光纤光缆（新增）
            "601869": "长飞光纤（光纤光缆龙头）",
            "600487": "亨通光电（光缆/海缆）",
            "600522": "中天科技（光缆/海缆）",
        },
    },
}


def get_all_us_tickers() -> list:
    tickers = set()
    for s in SECTORS.values():
        tickers.update(s["us"].keys())
    return sorted(tickers)


def get_all_kr_tickers() -> list:
    tickers = set()
    for s in SECTORS.values():
        tickers.update(s["kr"].keys())
    return sorted(tickers)


def get_all_ashare_codes() -> dict:
    """返回 {code: name} 的全量A股字典"""
    codes = {}
    for s in SECTORS.values():
        codes.update(s["a"])
    return codes


def get_sector_for_code(code: str) -> "str | None":
    """返回某A股code所属的板块key"""
    for sk, sd in SECTORS.items():
        if code in sd["a"]:
            return sk
    return None

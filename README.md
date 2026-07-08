# 🛡️ AI-Quant: China-US-Korea GPU LeadLag Predictor

![Version](https://img.shields.io/badge/Version-v6.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9+-brightgreen.svg)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-orange.svg)
![Contributor](https://img.shields.io/badge/Contributor-@ywan4454-red.svg)

> **基于“中韩美跨市场重力传导”与“特征隔离惩罚”的 A 股 GPU/AI 产业链早盘量化预测系统。**

本系统通过追踪美股和韩股核心科技巨头（如英伟达、AMD、ASML、SK 海力士等）的最新价格动态，利用严格的交易时差优势，在 A 股开盘前自动输出 AI 产业链 5 大核心板块及 22 只龙头个股的当日多空趋势，为您提供具备**实战风控防爆器（盾牌 🛡️）**属性的量化交易指令。

---

## 🧠 核心逻辑与算法引擎 (The Logic)

全球半导体和 AI 产业链高度联动，但各大资本市场存在显著的交易时差。利用这种时差，我们可以建立一条单向的数据传导链：
1. **🇺🇸 美股 (T-1 日收盘)**：全球 AI 风向标，其收盘时间在 A 股 T 日开盘之前。
2. **🇰🇷 韩股 (T 日 10:00 KST)**：全球存储 (HBM) 与先进制造重镇，开盘比 A 股早 1 个小时，反映隔夜情绪的初步消化。
3. **🇨🇳 A股 (T 日 09:30 CST)**：全球重要的 AI 算力应用、光模块及代工制造基地（预测目标）。

### ⚙️ 核心算法机制 (Algorithm Pipeline)
1. **🛡️ 跨市场特征隔离 (Feature Isolation)**:
   - 将 A 股 AI 产业链划分为 5 大生态板块独立建模。
   - 实施**严格的物理特征隔离**，各板块模型只“看”与其相关的上游或对标外盘因子，防止数据泄露（例如，光模块板块不会受到半导体设备数据噪音的污染）。
2. **🛡️ 重力惩罚机制 (Gravity Penalty)**:
   - 使用 **Random Forest 三分类模型**，将目标收益映射为看多(>1%)、看空(<-1%)与中性(震荡)。
   - 创新性引入**错判趋势惩罚系数 (5.0x)**：通过设置 `class_weight={1: 5.0, -1: 5.0, 0: 1.0}`，实施重拳惩罚，强制模型对确定性高的机会作出非中性的果断判断，极大提升了模型对大涨大跌的灵敏度。
3. **🛡️ 多维特征工程 (Advanced Feature Engineering)**:
   - **TAM 市场规模加权**：基于各细分行业的 2026 年 TAM 天花板，对对标龙头进行加权。
   - **短期平滑动量 (3D Momentum)**：引入 3 日滑动均值收益，识别稳定动量而非单日随机噪音。
4. **🛡️ 状态过滤 (Regime Shift)**:
   - 严格截取 **2025年9月** 之后（全球 AI 与存储产业逻辑高度共振后）的市场数据作为训练集，确保捕捉当前最真实的微观联动规律。

> **💡 真实回测表现 (基于全样本评估)**:
> 引入特征隔离与重力惩罚后，各板块全样本准确率大幅跑赢随机概率（在严苛的 ±1% 阈值下）：
> - `板块1·设计生态`：56.12% 
> - `板块2·制造生态`：54.84%
> - `板块3·封装互联`：53.76%
> - `板块4·整机与部署`：55.10%
> - `板块5·连接与散热`：58.16%

---

## 📡 覆盖的 5 大核心板块 (Sectors)

为了提高预测准确率，系统将 22 只龙头标的划分为 5 大生态板块进行集群预测：

1. **板块1·设计生态** (EDA / GPU / CPU / NPU)
   - *对标*: 寒武纪、景嘉微、龙芯中科
2. **板块2·制造生态** (晶圆代工 / 光刻设备 / HBM内存)
   - *对标*: 中芯国际、北方华创、中微公司、沪硅产业
3. **板块3·封装互联** (CoWoS先进封装 / ABF基板 / MSAP高端PCB)
   - *对标*: 长电科技、通富微电、华天科技、生益电子等
4. **板块4·整机与部署** (AI服务器 ODM / 数据中心)
   - *对标*: 浪潮信息、中科曙光
5. **板块5·连接与散热** (光模块 / 液冷 / 连接器 / 光缆)
   - *对标*: 中际旭创、天孚通信、新易盛、亨通光电等

---

## 🛠️ 部署与运行 (Usage)

本系统采用单一命令行入口 (`main.py`)，支持多种功能模式。数据底层依托于 `yfinance` 实盘抓取。

```bash
# 1. 克隆代码到本地
git clone https://github.com/ywan4454/China-US-Korea-GPU-LeadLagPredictor.git
cd China-US-Korea-GPU-LeadLagPredictor

# 2. 安装环境依赖
pip install -r requirements.txt

# 3. 运行日常早盘预测 (推荐每日 09:00 运行)
# ⚠️ 提示：国内运行请确保网络环境可畅通访问 Yahoo Finance
source venv/bin/activate
python main.py
```

---

## 🤝 贡献者 (Contributors)

* 🎖️ **核心贡献者 / 架构设计**: [@ywan4454](https://github.com/ywan4454)

欢迎大家为这个开源项目贡献代码！请按照以下标准工作流提交：
1. **Fork 本仓库** 到你的账号下。
2. **创建分支**: `git checkout -b feature/your-feature`
3. **提交代码**: `git commit -m 'Add new AI module'`
4. **推送到你的仓库**: `git push origin feature/your-feature`
5. **发起 Pull Request (PR)**: 我们会认真审阅每一行代码！

---
* **🛡️ 终极免责声明 (Disclaimer)**: 本系统基于量化统计模型，历史规律绝不代表未来表现。模型仅截取客观的价格量价与时序信号，未考虑基本面突变、地缘政治、突发事件等黑天鹅因素。所有的 AI 预测结果仅供技术交流与代码参考，**绝对不构成任何实质性的投资建议，股市有风险，入市需谨慎**。*

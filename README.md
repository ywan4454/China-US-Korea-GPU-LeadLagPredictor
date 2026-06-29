# China-US-Korea GPU LeadLag Predictor (GPU/AI 产业链跨市场预测系统)

**AI-Quant System**: 基于跨市场 Lead-Lag（领涨-跟涨）效应的 A 股 GPU/AI 产业链早盘预测系统。

本系统通过追踪美股和韩股核心科技巨头（如英伟达、AMD、ASML、SK 海力士等）的最新价格动态，利用时差优势，在 A 股开盘前自动输出 AI 产业链 5 大核心板块及 22 只龙头个股的当日上涨概率。

## 核心逻辑 (The Logic)

全球半导体和 AI 产业链高度联动，但各大资本市场存在显著的交易时差：
1. **美股 (T-1 日)**：全球 AI 风向标，其收盘时间在 A 股 T 日开盘之前。
2. **韩股 (T 日早盘)**：全球存储 (HBM) 与制造重镇，其开盘时间比 A 股早 1 个小时。
3. **A股 (T 日)**：全球重要的 AI 算力应用、光模块及代工制造基地。

**核心算法机制**：
- 提取 **美股 T-1 日的收盘收益率** 作为预测因子。
- 提取 **韩股 T 日早盘 10:00 (KST) 的截面收益率** `(Price_10:00 - Close_{T-1}) / Close_{T-1}` 作为预测因子，以精准捕捉韩股开盘后 1 小时的真实资金博弈情绪。
- 利用上述前置海外因子，通过机器学习模型对 A 股对应映射标的进行二元分类预测（当日是否上涨），并输出上涨概率 (PROB_UP)。
- 系统每天早盘 (北京时间 09:10) 自动通过 GitHub Actions 运行，并通过企业微信 Webhook 按照极客风格 (Geek Style) 推送当日预测报告及过去 7 天的回测胜率。

### 智能信号过滤与评估 (Smart Signal Filtering)
系统对每个板块进行了独立的“最优中性阈值（Threshold, th）”搜索，保证仅在胜率最高且信号充分时才发出预测：
- **`th` (最优阈值)**: 模型自动在回测中寻找的置信区间。例如 `th=0.62` 意味着只有当上涨概率 > 62% 才看涨，< 38% 才看跌。
- **`[UP] / [DOWN]`**: 概率突破阈值，触发明确的看涨/看跌信号。
- **`[NEUTRAL]`**: 概率落在阈值区间内，系统判定为**中性（不出手）**，过滤掉掷硬币行情。
- **`WR` (Filtered Win Rate)**: 过滤掉中性信号后，该板块在整个测试集（而非仅最近 7 天）中的实际“出手胜率”。
- **`PAST_7` 符号说明**:
  - `✅`: 预测正确
  - `❌`: 预测错误
  - `➖`: 信号为中性（未出手，不计入胜负）
  - `⚪`: 尚无真实数据验证


## 数据来源 (Data Sources)

所有量化数据均通过 **`yfinance` (Yahoo Finance)** 接口抓取。回测训练集严格截取自 **2025年9月**，以框定 AI/存储 爆发后的强相关性市场结构（Regime Shift），剔除远期无效噪音。

### 1. 前置特征数据 (Features)
- **美股 (US Stocks)**: NVDA, AMD, ASML, AMAT, CDNS, SMCI, COHR 等 16 只产业链核心美股。
- **韩股 (Korean Stocks)**: 005930.KS (三星电子), 000660.KS (SK Hynix), 009150.KS (三星电机) 等 3 只核心韩股。

### 2. 预测目标数据 (Targets)
- **A股 (A-Shares)**: 选取《全球GPU与AI算力产业链全景图》中 15 大环节的 22 只核心 A 股标的。

## 覆盖的 5 大核心板块 (Sectors)

为了提高预测准确率，系统将标的划分为 5 大生态板块进行集群预测：

1. **板块1·设计生态** (EDA / GPU / CPU / NPU)
   - *对标*: 寒武纪、景嘉微、龙芯中科
2. **板块2·制造生态** (晶圆代工 / 光刻设备 / HBM内存)
   - *对标*: 中芯国际、北方华创、中微公司、沪硅产业
3. **板块3·封装互联** (CoWoS先进封装 / ABF基板 / MSAP高端PCB)
   - *对标*: 长电科技、通富微电、华天科技等
4. **板块4·整机与部署** (AI服务器 ODM / 数据中心)
   - *对标*: 浪潮信息、中科曙光
5. **板块5·连接与散热** (光模块 / 液冷 / 连接器 / 光纤)
   - *对标*: 中际旭创、天孚通信、新易盛等

## 部署与运行 (Usage)

本系统支持两种运行模式，你可以根据需求选择：

### 模式一：云端自动化推送（推荐 / 零代码配置）
本项目配置了 GitHub Actions CI/CD 流水线，可实现每日全自动预测并推送到你的手机。

1. 点击右上角 **Fork** 将本仓库复制到你的 GitHub 账号下。
2. 在你的仓库中进入 `Settings` -> `Secrets and variables` -> `Actions`。
3. 新建 Secret：名称填入 `WECHAT_WEBHOOK_URL`，值填入你的企业微信群机器人的 Webhook 链接。
4. 每天工作日北京时间 **09:10 (UTC 01:10)**，GitHub 服务器将自动运行模型并为你推送预测结果。

### 模式二：极客本地运行模式
如果你喜欢钻研量化代码，可以直接在本地 Bash 终端中运行：

```bash
# 1. 下载代码到本地
git clone https://github.com/ywan4454/China-US-Korea-GPU-LeadLagPredictor.git

# 2. 进入项目目录
cd China-US-Korea-GPU-LeadLagPredictor

# 3. 安装必备依赖 (请确保已安装 Python)
pip install -r requirements.txt

# 4. 运行预测模型
# 提示：如在国内运行，因访问 Yahoo Finance，可能需要配置终端代理，例如：
# export https_proxy=http://127.0.0.1:7890
python main.py
```

## 如何贡献代码 (How to Contribute)

欢迎大家为这个开源项目贡献代码！由于安全权限控制，你不能直接将代码 push 到本仓库。请按照以下标准的开源工作流提交你的代码：

1. **Fork 本仓库**：点击右上角的 `Fork` 按钮，将项目复制到你的 GitHub 账号下。
2. **克隆到本地**：`git clone https://github.com/你的用户名/China-US-Korea-GPU-LeadLagPredictor.git`
3. **创建分支**：`git checkout -b feature/your-feature-name`
4. **提交代码**：`git commit -m 'Add some amazing feature'`
5. **推送到你的仓库**：`git push origin feature/your-feature-name`
6. **发起 Pull Request (PR)**：回到本仓库主页，点击 `Pull requests` -> `New pull request`，提交你的修改请求。我会在这里手动审阅你的代码，审核通过后就会合并到主分支中！

---
* **免责声明 (Disclaimer)**: 本系统基于量化统计模型，历史规律不代表未来表现。模型仅使用价格量价信号，未考虑基本面、政策、突发事件等因素。预测结果仅供技术交流与参考，不构成任何实质性投资建议。*

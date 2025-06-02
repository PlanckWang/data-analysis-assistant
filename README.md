# 数据分析聊天助手 (Data Analysis Chatbot)

基于 MCP (Model Context Protocol) 的智能数据分析助手，支持通过自然语言与 ChatGPT、Claude、DeepSeek 等大语言模型进行交互，实现数据分析和可视化。

## 功能特性

- 🤖 **多模型支持**: 支持 ChatGPT、Claude、DeepSeek 等主流 LLM
- 📊 **数据分析**: 数据概览、描述性统计、相关性分析等
- 📈 **数据可视化**: 直方图、散点图、箱线图、热力图、时间序列图等
- 🔬 **统计检验**: 正态性检验、t检验、ANOVA、卡方检验等
- 💬 **自然语言交互**: 通过对话方式进行数据分析
- 📁 **文件支持**: 支持 CSV、Excel 格式数据文件
- 🌐 **Web 界面**: 现代化的聊天式 Web 界面
- 🛠️ **LLM 工具调用**: LLM 通过结构化的函数调用与后端数据分析工具交互，执行复杂任务。

## 快速开始

### 1. 安装依赖

```bash
cd /Users/derrick/Documents/数据分析/data_analysis
pip install -e .
```

### 2. 配置环境变量

复制 `.env.example` 文件为 `.env` 并填入相应的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件，添加你的 API 密钥：

```env
# OpenAI (ChatGPT)
OPENAI_API_KEY=你的OpenAI API密钥

# Anthropic (Claude)
ANTHROPIC_API_KEY=你的Anthropic API密钥

# DeepSeek
DEEPSEEK_API_KEY=你的DeepSeek API密钥

# 默认使用的模型
DEFAULT_LLM_PROVIDER=openai
```

### 3. 启动服务

```bash
# 启动聊天机器人界面
python -m src.chatbot_app

# 或者使用传统的 Web 界面
python -m src.web_app
```

访问 http://localhost:8000 开始使用。

## 使用示例

1. **上传数据文件**
   - 点击或拖拽 CSV/Excel 文件到上传区域
   - 支持的格式：.csv, .xlsx, .xls

2. **数据分析对话示例**
   - "请给我展示数据的基本信息"
   - "分析销售额和利润之间的相关性"
   - "画一个价格分布的直方图"
   - "对不同地区的销售额进行方差分析"
   - "检验数据是否符合正态分布"

3. **切换 AI 模型**
   - 在左侧边栏选择不同的 AI 模型
   - 支持实时切换，保持对话上下文

**重要提示**: 上传的数据和执行的分析是会话特定的，如果服务器重新启动则不会保留，因为数据当前存储在内存中。

## 项目结构

```
data_analysis/
├── src/
│   ├── server.py          # MCP 服务器实现
│   ├── chatbot_app.py     # 聊天机器人 Web 应用
│   ├── web_app.py         # 传统 Web 界面（可选）
│   ├── llm_providers.py   # LLM 提供商实现
│   ├── visualization.py   # 数据可视化工具
│   └── statistical_tests.py # 统计检验工具
├── templates/
│   └── chatbot.html       # 聊天机器人界面 HTML
├── static/
│   ├── css/               # CSS 样式 (如果未来添加)
│   └── js/
│       └── chatbot_scripts.js # 聊天机器人界面的客户端 JavaScript
├── uploads/              # 上传文件目录
├── exports/              # 导出文件目录
├── .env.example          # 环境变量示例
├── .gitignore           # Git 忽略文件
├── pyproject.toml       # 项目配置
└── README.md            # 本文件
```

## 前端说明

聊天机器人的主要客户端逻辑位于 `static/js/chatbot_scripts.js`。此文件处理用户交互、与后端 API 通信以及在浏览器中呈现聊天消息和数据可视化。

## 高级功能

### LLM 工具调用 (Function Calling)
本应用的一个核心特性是LLM能够调用后端定义好的数据分析工具。这是通过MCP (Model Context Protocol) 实现的，LLM会生成特定格式的指令来请求工具执行，并将结果用于后续的分析和回答。这使得LLM能够执行诸如文件上传处理、统计计算和图表生成等复杂操作。

### 统计检验工具

- **正态性检验**: Shapiro-Wilk、Kolmogorov-Smirnov、Anderson-Darling 等
- **均值比较**: 单样本t检验、双样本t检验、配对t检验
- **方差分析**: 单因素 ANOVA，包含事后检验
- **相关性检验**: Pearson、Spearman、Kendall 相关系数及显著性
- **独立性检验**: 卡方检验用于分类变量

### 可视化选项

- 支持 Matplotlib 静态图表
- 支持 Plotly 交互式图表
- 自动选择合适的图表类型
- 可导出图表为图片格式

## 注意事项

1. 确保已安装 Python 3.10 或更高版本
2. API 密钥需要有效且有足够的配额
3. 大文件上传可能需要较长处理时间
4. 建议定期导出重要的分析结果
5. **数据持久性**: 上传的数据和执行的分析是会话特定的，如果服务器重新启动则不会保留，因为数据当前存储在内存中。 (Data uploaded and analyses performed are session-specific and are not persisted if the server restarts, as data is currently stored in memory.)
6. **代码库状态**: 代码库最近进行了一些重构和改进，包括前端JavaScript的组织和LLM提供者逻辑的清理，以提高可维护性。

## 故障排除

### 常见问题

1. **无法连接到 MCP 服务器**
   - 检查 Python 环境是否正确
   - 确保所有依赖都已安装

2. **API 调用失败**
   - 检查 API 密钥是否正确
   - 确认网络连接正常
   - 检查 API 配额是否充足

3. **文件上传失败**
   - 检查文件格式是否支持
   - 确认文件大小不超过限制
   - 检查文件编码（建议使用 UTF-8）

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
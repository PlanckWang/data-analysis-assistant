# 数据分析助手开发文档

## 项目概述

基于 MCP (Model Context Protocol) Python SDK 开发的本地数据分析助手，支持 CSV 文件上传、数据分析和可视化功能。

## 技术架构

### 核心技术栈
- **后端框架**: FastMCP (参考 MCP Python SDK, 文件夹在 /Users/derrick/Documents/数据分析/python-sdk)
- **数据处理**: pandas, numpy等包
- **可视化**: matplotlib, seaborn, plotly等包
- **Web框架**: FastAPI + Starlette
- **前端**: HTML5 + JavaScript + Bootstrap
- **文件处理**: python-multipart

### 项目结构
项目在/Users/derrick/Documents/数据分析/data_analysis 中进行开发

## Web 界面设计
1. 一个典型的chat webui，用户通过上传文件和与模型进行对话来进行数据分析
2. 支持图表、数据分析结果的导出
3. 可查看历史对话记录

### 分析功能
1. **数据概览**: 数据形状、类型、缺失值
2. **统计分析**: 支持多种从简单到复杂的描述性统计、相关性分析，
3. **统计学检验**：确保分析结果符合所使用方法要求的统计学检验
4. **可视化面板**: 交互式图表生成


## 开发规范

### 代码风格
- 使用 Python 类型注解
- 遵循 PEP 8 代码规范
- 函数和类添加详细文档字符串
- 使用 ruff 进行代码格式化

### 错误处理
- 所有工具函数必须包含异常处理
- 返回标准化的错误信息格式
- 记录详细的错误日志

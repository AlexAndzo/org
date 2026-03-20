# Call Match Package 使用文档

> 通话质量分析系统 - 用于评估 SDR（销售开发代表）通话质量的自动化工具

---

## 功能概述

本包包含两个核心功能：

| 步骤 | 脚本 | 功能 |
|------|------|------|
| **Step 1** | `csv_to_call_opening.py` | 从 CSV 导入开场白模板到数据库 |
| **Step 2** | `analyze_calls.py` | 分析通话记录，计算三个匹配分数 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env 填入实际值
```

### 3. 运行脚本

```bash
# Step 1: 导入开场白
python csv_to_call_opening.py

# Step 2: 分析通话
python analyze_calls.py
```

---

## 环境变量说明

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `DB_HOST` | 是 | localhost | 数据库主机 |
| `DB_PORT` | 否 | 3306 | 数据库端口 |
| `DB_USER` | 是 | root | 数据库用户名 |
| `DB_PASSWORD` | 是 | - | 数据库密码 |
| `DB_NAME` | 是 | sdr_backend | 数据库名 |
| `LLM_API_KEY` | 是 | - | OpenAI API Key |
| `LLM_BASE_URL` | 否 | https://api.openai.com/v1 | API 地址 |
| `LLM_MODEL` | 否 | gpt-4o-mini | 模型名称 |
| `CSV_FILE` | 否 | ./openings.csv | CSV 文件路径 |

---

## 配置业务内容

编辑 `config.py` 中的以下内容以适配你的业务：

### 预设问题与答案

```python
QUESTIONS_ANSWERS = [
    {
        "question": "你的第一个问题?",
        "answers": ["答案1", "答案2", "答案3"],
    },
]
```

### 话术脚本

```python
FOLLOW_UP_SCRIPT = """
你的 SDR 话术脚本内容...
"""
```

---

## 数据流程

```
1. 从 CSV 读取公司开场白
   ↓
2. 写入数据库 call_opening 字段
   ↓
3. 分析通话记录
   ↓
4. 计算三个匹配分数
   ↓
5. 写回数据库
```

---

## 输出字段说明

| 字段 | 含义 | 评估对象 |
|------|------|----------|
| `opening_match` | 开场白匹配度 (0-100) | Dialer |
| `question_match` | 问题使用率 (0-100) | Dialer |
| `answer_match` | 答案匹配度 (0-100) | Receiver |

---

## 注意事项

1. 所有敏感信息（数据库密码、API Key）必须通过环境变量配置
2. 话术脚本和问题答案请根据实际业务需求修改
3. 首次使用前请确保数据库表结构正确

---

## 许可证

MIT License

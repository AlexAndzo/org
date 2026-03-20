"""
配置模块 - 通话质量分析系统

编辑说明:
- 敏感配置（数据库密码、API Key）从环境变量读取
- 业务配置（问题、答案、开场白脚本）请根据实际业务需求修改

环境变量:
- DB_HOST: 数据库主机
- DB_PORT: 数据库端口 (默认: 3306)
- DB_USER: 数据库用户名
- DB_PASSWORD: 数据库密码
- DB_NAME: 数据库名
- LLM_API_KEY: LLM API Key
- LLM_BASE_URL: LLM API 地址 (默认: https://api.openai.com/v1)
- LLM_MODEL: 模型名称 (默认: gpt-4o-mini)
- CSV_FILE: CSV文件路径
"""

import os
from pathlib import Path


def _get_env(key: str, default: str = "") -> str:
    """从环境变量获取配置，支持默认值"""
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    """从环境变量获取整数配置"""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """从环境变量获取浮点数配置"""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# ---------------- 数据库配置 ----------------
DB_CONFIG = {
    "host": _get_env("DB_HOST", "localhost"),
    "port": _get_env_int("DB_PORT", 3306),
    "user": _get_env("DB_USER", "root"),
    "password": _get_env("DB_PASSWORD", ""),
    "database": _get_env("DB_NAME", "sdr_backend"),
    "cursorclass": None,  # 由脚本内部设置
    "charset": "utf8mb4",
    "connect_timeout": 30,
    "read_timeout": 60,
    "write_timeout": 60,
}

# ---------------- CSV 文件路径 ----------------
# 默认使用当前目录下的 openings.csv，可通过环境变量覆盖
DEFAULT_CSV = str(Path(__file__).resolve().parent / "openings.csv")
CSV_FILE = _get_env("CSV_FILE", DEFAULT_CSV)

# ---------------- 预设问题与答案 ----------------
# 结构：列表中每个元素是一个 dict，包含：
#   question: 必填，字符串
#   answers:  可选，字符串列表；为空或缺省表示没有标准答案
#
# 请根据实际业务需求修改以下内容
QUESTIONS_ANSWERS = [
    {
        "question": "What does your typical [service/work] process look like?",
        "answers": [
            "Process A",
            "Process B",
            "Process C",
            "Not applicable",
        ],
    },
    {
        "question": "What is your biggest challenge with [topic]?",
        "answers": [
            "Challenge 1",
            "Challenge 2",
            "Challenge 3",
            "Not sure",
        ],
    },
    {
        "question": "How do you currently handle [issue]?",
        "answers": [],
    },
    {
        "question": "Have you considered using [solution]?",
        "answers": [],
    },
    {
        "question": "What would an ideal solution look like for you?",
        "answers": ["Answer 1", "Answer 2", "Answer 3"],
    },
    {
        "question": "Who is involved in the decision-making process?",
        "answers": [
            "Solo decision maker",
            "Team / committee",
            "Automated / policy-driven",
        ],
    },
    {
        "question": "Would you be interested in learning more about how we can help?",
        "answers": [
            "Yes - definitely interested",
            "Yes - somewhat interested",
            "Maybe - need more information",
            "Not a priority right now",
        ],
    },
]

# ---------------- LLM 配置 ----------------
LLM_CONFIG = {
    "api_key": _get_env("LLM_API_KEY", ""),
    "model": _get_env("LLM_MODEL", "gpt-4o-mini"),
    "base_url": _get_env("LLM_BASE_URL", "https://api.openai.com/v1"),
    "temperature": _get_env_float("LLM_TEMPERATURE", 0.2),
}

# ---------------- 阈值配置 ----------------
USAGE_THRESHOLD = _get_env_int("USAGE_THRESHOLD", 50)    # 问题使用率阈值
OPENING_THRESHOLD = _get_env_int("OPENING_THRESHOLD", 70)  # 开场白匹配阈值
ANSWER_THRESHOLD = _get_env_int("ANSWER_THRESHOLD", 70)   # 答案匹配阈值

# ---------------- 后续话术脚本 ----------------
# SDR 应该遵循的完整话术脚本（call_opening + FOLLOW_UP_SCRIPT）
# 请根据实际业务需求修改以下内容
#
# 脚本结构说明：
# 1. 开场白：介绍自己和公司，提及从网站/渠道了解到对方
# 2. 痛点探索：询问对方当前面临的挑战
# 3. 方案介绍：介绍自己的解决方案
# 4. 预约邀请：邀请对方进一步沟通
FOLLOW_UP_SCRIPT = """
Hi, this is [Your Name] from [Your Company].

I came across [Company Name]'s website and noticed you specialize in [their specialty].

I work with companies like yours who are facing [common challenge]. Have you encountered similar issues?

[Pause for response]

I hear really different things depending on who I talk to.

For some folks, it's [challenge A]. For others, it's more about [challenge B].

Where does that land for you?

[Pause for response]

What does your current process for [related task] look like?

[Pause for response]

Have you ever considered using [alternative approach]?

Or has that never really come up for you?

And just so I'm respectful of your time — when it comes to [key topic], are you usually the one making those decisions, or does someone else weigh in?

Here's what we've been learning from similar companies.

[Key insight about the problem and your solution approach]

We don't replace your existing tools or workflows. We sit alongside them and help you [specific benefit].

Instead of me trying to explain everything quickly over the phone, the best way is to walk through it together.

We usually do a short 15 to 30 minute working call, a few days out — typically within the next few days.

Would later this week or early next week work better for you?
"""

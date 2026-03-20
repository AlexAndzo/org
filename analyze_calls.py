"""
步骤2-6：分析通话记录并写回 opening_match / question_match / answer_match

流程：
1) 过滤 date >= 2026-02-02 且 transcription 非空
2) 解析 transcription（列表格式，A/B 双方对话），识别 dialer / receiver
3) 与 call_opening 比对，写 opening_match (百分比 int)
4) 与 QUESTIONS_ANSWERS 的 questions 比对，写 question_match (使用率 int)
5) 对已提问且有答案的问题，计算回答匹配度，写 answer_match (百分比 int)

数据格式：
- transcription 字段为 JSON 列表，每项格式如 "A : text" 或 "B : text"
- A 和 B 分别表示通话双方（B 通常是 dialer/agent，A 通常是 receiver/contact）

可编辑：
- DB 配置、Questions/Answers: call_match_package/config.py
"""

import asyncio
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymysql
from loguru import logger
from openai import AsyncOpenAI

import config

# Configure file logging - logs to call_match_package/logs directory
LOGS_DIR = Path(__file__).parent / "logs"

# Try to set up file logging, but don't crash if we don't have permission
try:
    LOGS_DIR.mkdir(exist_ok=True)
    logger.add(
        LOGS_DIR / "step2_analyze_calls_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        encoding="utf-8",
    )
    logger.info(f"[STEP2] File logging enabled: {LOGS_DIR}")
except PermissionError as e:
    logger.warning(
        f"[STEP2] Cannot write to log directory {LOGS_DIR}: {e}. Logs will only appear in console."
    )
except Exception as e:
    logger.warning(
        f"[STEP2] Failed to set up file logging: {e}. Logs will only appear in console."
    )


DATE_CUTOFF = date(2026, 2, 2)
QUESTION_SIM_THRESHOLD = 0.6  # 问题匹配阈值（规则回退用）


@dataclass
class CallRecord:
    id: Any
    call_opening: Optional[str]
    transcription: str  # JSON list format: ["A : text", "B : text", ...]
    date: Any
    talking_time: Optional[int]
    real_person: Optional[bool]
    contact_company: Optional[str] = None  # For logging purposes
    # Existing match scores (for incremental calculation)
    existing_opening_match: Optional[int] = None
    existing_question_match: Optional[int] = None
    existing_answer_match: Optional[int] = None


def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _similarity(a: str, b: str) -> float:
    a_norm, b_norm = _normalize(a), _normalize(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _best_similarity(text: str, candidates: List[str]) -> float:
    return max((_similarity(text, c) for c in candidates), default=0.0)


# ---------------- LLM 帮助 ----------------
_llm_client: Optional[AsyncOpenAI] = None


def _get_llm():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    api_key = config.LLM_CONFIG.get("api_key")
    if not api_key:
        return None
    _llm_client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.LLM_CONFIG.get("base_url"),
    )
    return _llm_client


async def _ask_score(prompt: str) -> float:
    client = _get_llm()
    if not client:
        return 0.0
    try:
        resp = await client.chat.completions.create(
            model=config.LLM_CONFIG.get("model", "gpt-4o-mini"),
            temperature=config.LLM_CONFIG.get("temperature", 0.2),
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict grader. Output only a number between 0 and 100 (integer or float). No text, no explanation.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        return float(txt)
    except Exception as e:
        logger.warning(f"[LLM] scoring failed, fallback to 0.0 | err={e}")
        return 0.0


# ---------------- Embedding 语义分析 ----------------
_embedding_cache: Dict[str, List[float]] = {}
_EMBEDDING_MODEL = "text-embedding-3-large"
_SEMANTIC_CONTEXT_THRESHOLD = 0.6  # 语义相似度阈值


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算两个向量的余弦相似度。"""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _get_embedding(text: str) -> Optional[List[float]]:
    """
    获取文本的 embedding 向量，带内存缓存。
    使用与 LLM 相同的 API key。
    """
    global _embedding_cache
    text_stripped = text.strip()
    if not text_stripped:
        return None

    # 缓存命中
    cache_key = text_stripped[:200]  # 截断作为 key，避免过长
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    client = _get_llm()
    if not client:
        return None

    try:
        resp = await client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=text_stripped,
        )
        embedding = resp.data[0].embedding
        _embedding_cache[cache_key] = embedding
        return embedding
    except Exception as e:
        logger.warning(f"[EMBEDDING] Failed to get embedding: {e}")
        return None


async def _get_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    批量获取 embeddings，对已缓存的跳过 API 调用，未缓存的一次性批量请求。
    """
    global _embedding_cache
    results: List[Optional[List[float]]] = [None] * len(texts)
    uncached_indices: List[int] = []
    uncached_texts: List[str] = []

    for i, text in enumerate(texts):
        text_stripped = text.strip()
        if not text_stripped:
            continue
        cache_key = text_stripped[:200]
        if cache_key in _embedding_cache:
            results[i] = _embedding_cache[cache_key]
        else:
            uncached_indices.append(i)
            uncached_texts.append(text_stripped)

    if not uncached_texts:
        return results

    client = _get_llm()
    if not client:
        return results

    try:
        resp = await client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=uncached_texts,
        )
        for j, emb_data in enumerate(resp.data):
            idx = uncached_indices[j]
            embedding = emb_data.embedding
            cache_key = uncached_texts[j][:200]
            _embedding_cache[cache_key] = embedding
            results[idx] = embedding
        logger.debug(
            f"[EMBEDDING] Batch embedded {len(uncached_texts)} texts (cache size: {len(_embedding_cache)})"
        )
    except Exception as e:
        logger.warning(f"[EMBEDDING] Batch embedding failed: {e}")

    return results


async def _compute_semantic_context_score(
    full_segments: List[Dict[str, str]],
    dialer_speaker: str,
    receiver_speaker: str,
) -> Tuple[float, List[str]]:
    """
    使用 Embedding 计算 dialer→receiver 对话对的语义相似度，
    判断 receiver 是否在上下文中做出了相关回应。

    Returns:
        (score_boost, detail_reasons):
        - score_boost: 给真人分数的加分值
        - detail_reasons: 具体的匹配详情
    """
    if not full_segments or len(full_segments) < 2:
        return 0.0, []

    # 提取连续的 dialer→receiver 对话对
    pairs: List[Tuple[str, str]] = []  # (dialer_text, receiver_text)
    for i in range(1, len(full_segments)):
        prev_seg = full_segments[i - 1]
        curr_seg = full_segments[i]
        if (
            prev_seg.get("speaker") == dialer_speaker
            and curr_seg.get("speaker") == receiver_speaker
        ):
            dialer_text = prev_seg.get("text", "").strip()
            receiver_text = curr_seg.get("text", "").strip()
            if dialer_text and receiver_text and len(receiver_text.split()) >= 2:
                pairs.append((dialer_text, receiver_text))

    if not pairs:
        return 0.0, []

    # 批量获取所有文本的 embedding（去重 + 缓存）
    all_texts = []
    for d_text, r_text in pairs:
        all_texts.append(d_text)
        all_texts.append(r_text)

    embeddings = await _get_embeddings_batch(all_texts)

    # 计算每对的余弦相似度
    context_hits = 0
    detail_reasons: List[str] = []
    for pair_idx, (d_text, r_text) in enumerate(pairs):
        d_emb = embeddings[pair_idx * 2]
        r_emb = embeddings[pair_idx * 2 + 1]
        if d_emb is None or r_emb is None:
            continue
        sim = _cosine_similarity(d_emb, r_emb)
        if sim >= _SEMANTIC_CONTEXT_THRESHOLD:
            context_hits += 1
            # 截断显示，避免日志过长
            d_short = d_text[:50] + ("..." if len(d_text) > 50 else "")
            r_short = r_text[:50] + ("..." if len(r_text) > 50 else "")
            detail_reasons.append(
                f"embedding_match(sim={sim:.3f}, D='{d_short}', R='{r_short}')"
            )

    if context_hits == 0:
        return 0.0, []

    # 加分：每命中一对加 0.1，上限 0.3
    score_boost = min(0.3, 0.1 * context_hits)
    detail_reasons.insert(0, f"semantic_context_hits={context_hits}/{len(pairs)}")
    return score_boost, detail_reasons


def get_conn():
    cfg = config.DB_CONFIG.copy()
    cfg["cursorclass"] = pymysql.cursors.DictCursor
    return pymysql.connect(**cfg)


def _load_questions_answers() -> List[Dict[str, Any]]:
    items = []
    for item in config.QUESTIONS_ANSWERS:
        q = item.get("question")
        answers = item.get("answers") or []
        if q:
            items.append(
                {"question": q, "answers": [a for a in answers if a and str(a).strip()]}
            )
    return items


def _fetch_records() -> List[CallRecord]:
    """Fetch records including existing match scores for incremental calculation."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, call_opening, transcription, date, talking_time, real_person,
                       opening_match, question_match, answer_match, contact_company
                FROM sdr_cdr_inout_data
                WHERE transcription IS NOT NULL
                  AND date >= %s
                """,
                (DATE_CUTOFF,),
            )
            rows = cur.fetchall()
            return [
                CallRecord(
                    id=row["id"],
                    call_opening=row.get("call_opening"),
                    transcription=row["transcription"],
                    date=row.get("date"),
                    talking_time=row.get("talking_time"),
                    real_person=row.get("real_person"),
                    contact_company=row.get("contact_company"),
                    existing_opening_match=row.get("opening_match"),
                    existing_question_match=row.get("question_match"),
                    existing_answer_match=row.get("answer_match"),
                )
                for row in rows
            ]
    finally:
        conn.close()


def _identify_speakers(segments: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    智能识别对话双方的角色。
    通过分析对话内容特征判断谁是 dialer (agent/SDR) 和 receiver (contact/customer)。

    判断逻辑：
    1. SDR/Agent 通常会：
       - 主动介绍自己和公司 ("This is X from Y", "I'm calling from")
       - 说更多的话，尤其是开场和中间部分
       - 使用销售相关词汇 (product, service, solution, demo, meeting, schedule)
    2. Customer/Contact 通常会：
       - 回应问候 ("Hello", "Hi", "Yes")
       - 回答问题而非主动提问
       - 话语相对较短

    Args:
        segments: 解析后的对话片段列表，每个包含 {"speaker": "A"/"B", "text": "..."}

    Returns:
        (dialer, receiver): 识别出的 (SDR, Customer) 说话者标识
    """
    if not segments:
        return "B", "A"  # 默认值

    # 收集每个说话者的信息
    speakers = {}
    for seg in segments:
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").lower()
        if speaker not in speakers:
            speakers[speaker] = {
                "texts": [],
                "total_words": 0,
                "score": 0,  # 正分表示更可能是 SDR
            }
        speakers[speaker]["texts"].append(text)
        speakers[speaker]["total_words"] += len(text.split())

    if len(speakers) < 2:
        # 只有一个说话者，无法区分
        speaker_list = list(speakers.keys())
        if speaker_list:
            return speaker_list[0], speaker_list[0]
        return "B", "A"

    # SDR 特征关键词（出现则加分）
    sdr_indicators = [
        # 自我介绍
        "this is",
        "my name is",
        "i'm calling",
        "calling from",
        "i work with",
        "i work for",
        "we help",
        "we work with",
        # 销售词汇
        "solution",
        "product",
        "service",
        "demo",
        "meeting",
        "schedule",
        "appointment",
        "15 minute",
        "30 minute",
        "quick call",
        "touch base",
        "reach out",
        "follow up",
        # 提问方式（销售式提问）
        "what do you guys",
        "are you guys",
        "do you guys",
        "how do you",
        "is that correct",
        "does that make sense",
        "would you be interested",
        "can we schedule",
        "are you available",
    ]

    # Customer 特征关键词（出现则减分，说明更可能是 Customer）
    customer_indicators = [
        # 简短回应
        "yeah",
        "yes",
        "no",
        "nope",
        "okay",
        "ok",
        "sure",
        "right",
        "uh-huh",
        "mm-hmm",
        "uh huh",
        "mm hmm",
        # 询问身份（说明是被动接听）
        "who is this",
        "who's calling",
        "what company",
        # 拒绝/不感兴趣
        "not interested",
        "no thanks",
        "we're good",
        "we don't need",
        "already have",
        "we use",
        "happy with",
    ]

    # 计算每个说话者的分数
    for speaker, data in speakers.items():
        full_text = " ".join(data["texts"])

        # SDR 特征加分
        for indicator in sdr_indicators:
            if indicator in full_text:
                data["score"] += 2

        # Customer 特征减分
        for indicator in customer_indicators:
            if indicator in full_text:
                data["score"] -= 1

        # 话语量加分（SDR 通常说得更多）
        if data["total_words"] > 50:
            data["score"] += 1
        if data["total_words"] > 100:
            data["score"] += 1
        if data["total_words"] > 200:
            data["score"] += 1

        # 检查前几句话是否包含自我介绍（SDR 特征）
        first_texts = " ".join(data["texts"][:3])
        if "this is" in first_texts and "from" in first_texts:
            data["score"] += 3
        if "i'm" in first_texts and ("calling" in first_texts or "from" in first_texts):
            data["score"] += 3

    # 找出得分最高的说话者作为 SDR
    speaker_list = list(speakers.keys())
    if len(speaker_list) >= 2:
        speaker_list.sort(key=lambda s: speakers[s]["score"], reverse=True)
        dialer = speaker_list[0]  # 得分最高的是 SDR
        receiver = speaker_list[1]  # 得分次高的是 Customer

        logger.debug(
            f"[SPEAKERS] 识别结果: dialer={dialer} (score={speakers[dialer]['score']}), "
            f"receiver={receiver} (score={speakers[receiver]['score']})"
        )
        return dialer, receiver

    return "B", "A"


async def _rule_based_real_person_check(
    receiver_texts: List[str],
    dialer_texts: List[str] = None,
    full_segments: List[Dict[str, str]] = None,
) -> Tuple[Optional[bool], float, List[str]]:
    """
    基于规则的真人检测引擎。通过多种信号综合判断 receiver 是否为真人。

    Returns:
        (verdict, confidence, reasons):
        - verdict: True=真人, False=自动系统, None=不确定
        - confidence: 置信度 0.0~1.0
        - reasons: 判断依据列表（用于日志）
    """
    receiver_full = " ".join(receiver_texts).strip().lower()
    reasons: List[str] = []

    if not receiver_full:
        return None, 0.0, ["no_receiver_text"]

    # ========== 自动系统强信号 (AUTOMATED) ==========
    automated_phrases = [
        "leave a message",
        "leave your message",
        "leave me a message",
        "after the tone",
        "after the beep",
        "at the tone",
        "press 1",
        "press 2",
        "press 0",
        "press pound",
        "press star",
        "press the",
        "para español",
        "for english",
        "for spanish",
        "is not available",
        "is unavailable",
        "cannot take your call",
        "can't take your call",
        "unable to take your call",
        "not available right now",
        "please try again later",
        "please call back",
        "mailbox is full",
        "voicemail",
        "voice mail",
        "automated attendant",
        "your call is important",
        "your call has been forwarded",
        "the person you are trying to reach",
        "the number you have dialed",
        "the subscriber you have dialed",
        "please hold",
        "thank you for calling",  # 开头的自动问候
        "goodbye",  # 自动挂断
        "has not set up their voicemail",
        "record your message",
        "no one is available",
        "office hours",
        "business hours",
        "directory",
        "extension",
        "dial by name",
    ]

    automated_hits = []
    for phrase in automated_phrases:
        if phrase in receiver_full:
            automated_hits.append(phrase)

    if automated_hits:
        # 强自动信号 — 多个命中则更确信
        confidence = min(0.95, 0.7 + 0.1 * len(automated_hits))
        reasons.append(f"automated_phrases_found: {automated_hits}")
        return False, confidence, reasons

    # ========== 真人强信号 ==========
    real_person_score = 0.0
    total_signals = 0

    # --- 信号1: Receiver 提问（真人行为，机器不会问问题） ---
    question_patterns = [
        r"\bwho is this\b",
        r"\bwho'?s? (this|calling|there)\b",
        r"\bwhat company\b",
        r"\bwhat is this (about|regarding|for)\b",
        r"\bwhat do you (want|need|mean)\b",
        r"\bcan you (repeat|say that again|explain)\b",
        r"\bwhat'?s? (your|the) (name|number|company)\b",
        r"\bhow did you get (my|this) number\b",
        r"\bwhere (are you|did you) (calling|get)\b",
        r"\bsorry\s*,?\s*what\b",
        r"\bexcuse me\b.*\?",
        r"\bpardon\b",
        r"\breally\s*\?",
        r"\bwhy\b.*\?",
        r"\bhow (much|long|many|does|do|is|are|would|could)\b.*\?",
        r"\bwhat (kind|type|sort)\b",
    ]
    receiver_questions = 0
    for pat in question_patterns:
        if re.search(pat, receiver_full, re.IGNORECASE):
            receiver_questions += 1
    # 也检查句末问号
    for text in receiver_texts:
        if text.strip().endswith("?"):
            receiver_questions += 1
    if receiver_questions > 0:
        real_person_score += min(0.3, 0.15 * receiver_questions)
        reasons.append(f"asks_questions({receiver_questions})")
        total_signals += 1

    # --- 信号2: 对话轮次多样性（真人有多种不同回应） ---
    unique_responses = set()
    for text in receiver_texts:
        normalized = re.sub(r"[^\w\s]", "", text.strip().lower())
        if normalized and len(normalized) > 2:  # 忽略极短的
            unique_responses.add(normalized)
    distinct_turns = len(unique_responses)
    if distinct_turns >= 4:
        real_person_score += 0.3
        reasons.append(f"high_turn_diversity({distinct_turns})")
        total_signals += 1
    elif distinct_turns >= 2:
        real_person_score += 0.15
        reasons.append(f"moderate_turn_diversity({distinct_turns})")
        total_signals += 1

    # --- 信号3: 上下文回应（使用 Embedding 语义相似度判断） ---
    if dialer_texts and full_segments:
        # 识别 dialer 和 receiver 的 speaker 标识
        dialer_speaker = None
        receiver_speaker = None
        for seg in full_segments:
            sp = seg.get("speaker", "")
            txt = seg.get("text", "")
            if txt in dialer_texts and dialer_speaker is None:
                dialer_speaker = sp
            elif txt in receiver_texts and receiver_speaker is None:
                receiver_speaker = sp
            if dialer_speaker and receiver_speaker:
                break

        if dialer_speaker and receiver_speaker:
            emb_score, emb_details = await _compute_semantic_context_score(
                full_segments, dialer_speaker, receiver_speaker
            )
            if emb_score > 0:
                real_person_score += emb_score
                reasons.extend(emb_details)
                total_signals += 1
                logger.info(
                    f"[REAL_PERSON] Embedding context score: +{emb_score:.2f}, "
                    f"details: {emb_details}"
                )
            else:
                logger.debug("[REAL_PERSON] Embedding context: no significant matches")
        else:
            logger.debug(
                f"[REAL_PERSON] Could not identify speakers for embedding "
                f"(dialer={dialer_speaker}, receiver={receiver_speaker})"
            )

        # 回退：如果 embedding 不可用，使用关键词规则作为补充
        if not _get_llm():
            contextual_responses = 0
            for i, seg in enumerate(full_segments or []):
                if i == 0:
                    continue
                prev_segments = [
                    s
                    for s in (full_segments or [])[:i]
                    if s.get("speaker") != seg.get("speaker")
                ]
                if prev_segments:
                    curr_text = seg.get("text", "")
                    context_words = re.findall(
                        r"\b(yes|yeah|no|nope|we do|we don't|we have|we use|"
                        r"i think|i don't|actually|honestly|right now|currently|"
                        r"that's right|exactly|correct|good question|"
                        r"let me|i'll|we're|our|my)\b",
                        curr_text.lower(),
                    )
                    if context_words and len(curr_text.split()) >= 3:
                        contextual_responses += 1
            if contextual_responses >= 2:
                real_person_score += 0.2
                reasons.append(f"keyword_contextual_responses({contextual_responses})")
                total_signals += 1

    # --- 信号4: 自我介绍 / 提及身份 ---
    identity_patterns = [
        r"\b(this is|i'm|my name is|i am)\s+[A-Z][a-z]+",
        r"\bspeaking\b",
        r"\b(accounting|finance|office|project\s*manager|owner|secretary|receptionist)\b",
    ]
    for pat in identity_patterns:
        if re.search(pat, " ".join(receiver_texts), re.IGNORECASE):
            real_person_score += 0.15
            reasons.append("self_identification")
            total_signals += 1
            break

    # --- 信号5: 混合反应（短回应 + 长回应 = 自然对话节奏） ---
    short_responses = sum(1 for t in receiver_texts if len(t.split()) <= 3)
    long_responses = sum(1 for t in receiver_texts if len(t.split()) >= 6)
    if short_responses >= 1 and long_responses >= 1:
        real_person_score += 0.15
        reasons.append(
            f"mixed_response_lengths(short={short_responses}, long={long_responses})"
        )
        total_signals += 1

    # --- 信号6: 情感/人际互动词汇 ---
    emotional_patterns = [
        r"\b(oh|huh|hmm|wow|gosh|geez|man|well)\b",
        r"\b(interesting|appreciate|thank|thanks|sorry|sure|absolutely|definitely)\b",
        r"\b(i see|i understand|makes sense|sounds good|sounds like|tell me more)\b",
        r"\b(great|good|fine|okay|perfect|cool|nice|awesome)\b",
        r"\bha(ha)?\b",
        r"\blol\b",
    ]
    emotion_hits = 0
    for pat in emotional_patterns:
        if re.search(pat, receiver_full, re.IGNORECASE):
            emotion_hits += 1
    if emotion_hits >= 2:
        real_person_score += 0.15
        reasons.append(f"emotional_language({emotion_hits})")
        total_signals += 1

    # --- 信号7: 总字数和发言次数 ---
    total_words = sum(len(t.split()) for t in receiver_texts)
    total_turns = len(receiver_texts)
    if total_words >= 20 and total_turns >= 3:
        real_person_score += 0.1
        reasons.append(
            f"substantial_engagement(words={total_words}, turns={total_turns})"
        )
        total_signals += 1

    # ========== 综合判断 ==========
    if real_person_score >= 0.55 and total_signals >= 2:
        confidence = min(0.95, real_person_score)
        reasons.append(
            f"VERDICT=REAL (score={real_person_score:.2f}, signals={total_signals})"
        )
        return True, confidence, reasons
    elif real_person_score <= 0.1 and total_signals == 0 and total_words < 10:
        reasons.append(f"VERDICT=AMBIGUOUS_LOW (score={real_person_score:.2f})")
        return None, 0.3, reasons
    else:
        reasons.append(
            f"VERDICT=UNCERTAIN (score={real_person_score:.2f}, signals={total_signals})"
        )
        return None, real_person_score, reasons


async def _detect_real_person(
    receiver_texts: List[str],
    dialer_texts: List[str] = None,
    full_segments: List[Dict[str, str]] = None,
) -> bool:
    """
    判断 receiver 是否为真人（而非语音信箱、IVR 系统等）。
    采用三层策略：规则预检 → LLM 精判 → 规则兜底。
    返回 True 表示真人，False 表示非真人。
    """
    receiver_full = " ".join(receiver_texts).strip()
    if not receiver_full:
        logger.warning("[REAL_PERSON] No receiver text, assuming not real person")
        return False

    # ===== 第一层：规则预检 =====
    rule_verdict, rule_confidence, rule_reasons = await _rule_based_real_person_check(
        receiver_texts, dialer_texts, full_segments
    )
    logger.info(
        f"[REAL_PERSON] Rule check: verdict={rule_verdict}, confidence={rule_confidence:.2f}, "
        f"reasons={rule_reasons}"
    )

    # 高置信度规则判定 — 直接返回，跳过 LLM
    if rule_confidence >= 0.85 and rule_verdict is not None:
        logger.info(
            f"[REAL_PERSON] High-confidence rule verdict: {rule_verdict} "
            f"(confidence={rule_confidence:.2f}), skipping LLM"
        )
        return rule_verdict

    # ===== 第二层：LLM 精判 =====
    client = _get_llm()
    if not client:
        # 无 LLM — 使用规则结果
        logger.warning("[REAL_PERSON] No LLM configured, using rule-based result")
        if rule_verdict is not None:
            return rule_verdict
        # 规则也不确定，倾向于认为是真人（避免误杀）
        logger.warning(
            "[REAL_PERSON] Rules inconclusive, defaulting to REAL (benefit of doubt)"
        )
        return True

    # 构建完整对话上下文
    conversation_context = ""
    if full_segments:
        conversation_lines = []
        for seg in full_segments:
            speaker = seg.get("speaker", "?")
            text = seg.get("text", "")
            role_label = (
                "Dialer" if dialer_texts and text in dialer_texts else "Receiver"
            )
            conversation_lines.append(f"[{speaker}/{role_label}]: {text}")
        conversation_context = "\n".join(conversation_lines)
    else:
        conversation_context = (
            f"Receiver's speech (no full context available):\n{receiver_full}"
        )

    # 规则引擎给出的线索，供 LLM 参考
    rule_hint = ""
    if rule_reasons:
        rule_hint = (
            f"\n\nPreliminary rule-based analysis hints: {', '.join(rule_reasons)}"
        )

    try:
        prompt = f"""Analyze the following phone call transcription and determine if the RECEIVER (the person who answered the call) is a REAL PERSON or an AUTOMATED SYSTEM.

## Full Conversation:
{conversation_context}

## Receiver's text only:
{receiver_full}{rule_hint}

## Classification Guide:

REAL PERSON indicators (any of these strongly suggest a real person):
- Asks questions like "who is this?", "what company?", "what is this about?"
- Gives contextual replies that respond to what the dialer said
- Introduces themselves ("This is John", "speaking")
- Shows natural conversation flow: mix of short ("yeah", "okay") and longer responses
- Expresses opinions, emotions, hesitation, or confusion
- Has multiple varied turns of dialogue (not repeating the same phrase)
- Uses filler words naturally ("uh", "um", "well", "so", "like")
- Refers to their own company, role, or current situation

AUTOMATED SYSTEM indicators (multiple of these needed):
- "Leave a message after the tone/beep"
- "Press 1 for..., press 2 for..."
- "Your call is important to us / has been forwarded"
- "The person you are trying to reach is not available"
- "Please hold" or "please try again later"
- Formulaic greeting with no follow-up interaction
- Identical repeated phrases typical of a recording
- No variation in speech pattern; sounds scripted throughout

## Important Notes:
- A real person who gives SHORT answers (like "yes", "no", "who is this?") is still a REAL PERSON
- A receptionist or gatekeeper who screens calls is a REAL PERSON
- Someone who sounds annoyed, confused, or disinterested is still a REAL PERSON
- If the receiver engages in even minimal back-and-forth dialogue, they are very likely REAL
- ONLY classify as AUTOMATED if there are clear signs of a recording or menu system
- Your goal is ACCURACY, not strictness — when in doubt, lean toward REAL

Output format: First write either REAL or AUTOMATED, then a brief one-line reason.
Example: "REAL - Receiver asks clarifying questions and gives contextual responses"
Example: "AUTOMATED - Contains voicemail prompt asking caller to leave a message"
"""
        resp = await client.chat.completions.create(
            model=config.LLM_CONFIG.get("model", "gpt-4o-mini"),
            temperature=0.15,  # 低温度，稳定判断
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at analyzing phone call transcriptions to determine "
                        "if the call receiver is a real human or an automated system (voicemail, "
                        "IVR, answering machine). Your goal is ACCURACY — correctly identifying "
                        "real people is just as important as correctly identifying automated systems. "
                        "Many real people answer phone calls with short responses, sound confused, "
                        "or are gatekeepers — these are all REAL people. Only classify as AUTOMATED "
                        "when there is clear evidence of a recording, voicemail prompt, or phone menu."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        result = (resp.choices[0].message.content or "").strip()
        result_upper = result.upper()

        # 解析 LLM 结果 — 宽容匹配
        is_automated = any(
            kw in result_upper
            for kw in ["AUTOMATED", "VOICEMAIL", "IVR", "MACHINE", "RECORDING"]
        )
        is_real = any(kw in result_upper for kw in ["REAL", "HUMAN", "PERSON"])

        if is_automated and not is_real:
            llm_verdict = False
        elif is_real and not is_automated:
            llm_verdict = True
        elif is_real and is_automated:
            # 两个都出现了（比如 "REAL - not automated"），看哪个在前面
            real_pos = min(
                result_upper.find(kw)
                for kw in ["REAL", "HUMAN", "PERSON"]
                if kw in result_upper
            )
            auto_pos = min(
                result_upper.find(kw)
                for kw in ["AUTOMATED", "VOICEMAIL", "IVR", "MACHINE", "RECORDING"]
                if kw in result_upper
            )
            llm_verdict = real_pos < auto_pos  # 排在前面的是判断结果
        else:
            # 无法解析 — 使用规则结果
            llm_verdict = None

        logger.info(f"[REAL_PERSON] LLM result: '{result}' -> verdict={llm_verdict}")

        if llm_verdict is not None:
            return llm_verdict

        # LLM 输出无法解析 — 回退到规则
        logger.warning("[REAL_PERSON] LLM output ambiguous, falling back to rules")
        if rule_verdict is not None:
            return rule_verdict
        return True  # 最终兜底：给真人以benefit of the doubt

    except Exception as e:
        logger.warning(
            f"[REAL_PERSON] LLM detection failed, using rule-based fallback | err={e}"
        )
        # ===== 第三层：规则兜底 =====
        if rule_verdict is not None:
            logger.info(
                f"[REAL_PERSON] Fallback to rule verdict: {rule_verdict} "
                f"(confidence={rule_confidence:.2f})"
            )
            return rule_verdict
        # 规则也不确定 — 倾向真人
        logger.warning("[REAL_PERSON] All methods inconclusive, defaulting to REAL")
        return True


def _update_real_person(record_id: Any, is_real_person: bool):
    """更新数据库中的 real_person 字段。"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sdr_cdr_inout_data SET real_person = %s WHERE id = %s",
                (is_real_person, record_id),
            )
            conn.commit()
            logger.info(
                f"[REAL_PERSON] Updated record id={record_id} real_person={is_real_person}"
            )
    finally:
        conn.close()


def _parse_transcription(text: str) -> Tuple[Optional[Any], List[Dict[str, str]], None]:
    """
    解析 transcription 字段。

    新格式为 JSON 列表，每项格式如 "A : text" 或 "B : text"。
    例如: ["A : Hi, this is [Name].", "B : Hi, [Name]. This is [Your Name] from [Company].", ...]

    Args:
        text: transcription 字段的原始文本（JSON 字符串）

    Returns:
        (data, segments, None):
        - data: 原始解析后的数据
        - segments: 标准化的对话片段列表，每个包含 {"speaker": "A"/"B", "text": "..."}
        - None: 保持兼容性（原 callers 字段现在不再使用）
    """
    try:
        data = json.loads(text)
    except Exception as e:
        logger.warning(f"[PARSE] JSON 解析失败: {e}")
        return None, [], None

    segments = []

    if isinstance(data, list):
        # 新格式: ["A : text", "B : text", ...]
        for item in data:
            if isinstance(item, str):
                # 解析 "A : text" 或 "B : text" 格式
                if " : " in item:
                    parts = item.split(" : ", 1)
                    speaker = parts[0].strip()
                    text_content = parts[1].strip() if len(parts) > 1 else ""
                    segments.append({"speaker": speaker, "text": text_content})
                elif ":" in item:
                    # 备选解析："A: text" 格式（无空格）
                    parts = item.split(":", 1)
                    speaker = parts[0].strip()
                    text_content = parts[1].strip() if len(parts) > 1 else ""
                    segments.append({"speaker": speaker, "text": text_content})
            elif isinstance(item, dict):
                # 兼容旧格式: {"caller": "...", "text": "..."}
                segments.append(
                    {
                        "speaker": item.get("caller", "unknown"),
                        "text": item.get("text", ""),
                    }
                )
    elif isinstance(data, dict):
        # 兼容旧格式: {"segments": [...], "callers": []}
        old_segments = data.get("segments", [])
        for seg in old_segments:
            segments.append(
                {"speaker": seg.get("caller", "unknown"), "text": seg.get("text", "")}
            )

    return data, segments, None


async def _compute_opening(
    dialer_texts: List[str], call_opening: Optional[str], threshold: float
) -> Tuple[Optional[int], bool]:
    """
    计算 Opening Match（话术匹配度）

    新逻辑：比较 Dialer 的所有发言与完整话术脚本（call_opening + FOLLOW_UP_SCRIPT）的匹配程度。
    使用 LLM 进行语义匹配，优先保证精确度。

    Args:
        dialer_texts: Dialer 的所有发言文本列表
        call_opening: 该公司的开场白模板（可能为空）
        threshold: 匹配阈值

    Returns:
        (score, is_match): 匹配分数(0-100) 和 是否达到阈值
    """
    from loguru import logger

    # 构建完整话术脚本
    full_script_parts = []
    if call_opening and str(call_opening).strip():
        full_script_parts.append(str(call_opening).strip())

    # 添加后续话术
    follow_up = getattr(config, "FOLLOW_UP_SCRIPT", "")
    if follow_up and str(follow_up).strip():
        full_script_parts.append(str(follow_up).strip())

    if not full_script_parts:
        logger.debug(
            "[OPENING] 无话术脚本可比对（call_opening 和 FOLLOW_UP_SCRIPT 均为空）"
        )
        return None, False

    full_script = "\n\n".join(full_script_parts)

    # 获取 Dialer 的所有发言（不再只取前3句）
    if not dialer_texts:
        logger.debug("[OPENING] Dialer 无发言内容")
        return None, False

    dialer_full = " ".join(dialer_texts).strip()
    if not dialer_full:
        logger.debug("[OPENING] Dialer 发言内容为空")
        return None, False

    logger.info(
        f"[OPENING] Dialer 发言长度: {len(dialer_full)} 字符 | 脚本长度: {len(full_script)} 字符"
    )

    # 使用 LLM 进行语义匹配
    client = _get_llm()
    if client:
        try:
            prompt = f"""你是一个专业的销售话术分析师。请分析以下 SDR（销售开发代表）的实际通话内容与标准话术脚本的匹配程度。

## 标准话术脚本：
{full_script}

## SDR 实际发言内容：
{dialer_full}

## 评分标准：
- 100分：完全按照脚本说的，几乎一字不差
- 80-99分：遵循了脚本的主要内容和结构，有一些个人化表达
- 60-79分：覆盖了脚本的大部分关键点，但顺序或表达有较多变化
- 40-59分：只覆盖了部分脚本内容，遗漏了一些重要话术
- 20-39分：只涉及了脚本的少量内容
- 0-19分：几乎没有遵循脚本

请综合考虑以下因素：
1. 关键话术点的覆盖率（开场白、问题探索、解决方案介绍、预约邀请等）
2. 话术顺序是否合理
3. 核心概念是否传达到位（如核心关键词等）
4. 整体话术流程的完整性

请只输出一个 0-100 之间的整数分数，不要有任何其他文字。"""

            resp = await client.chat.completions.create(
                model=config.LLM_CONFIG.get("model", "gpt-4o-mini"),
                temperature=0.1,  # 低温度，更稳定的评分
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的销售话术匹配度评分专家。你只输出一个数字分数，不输出任何解释。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            result = (resp.choices[0].message.content or "").strip()
            # 提取数字
            import re

            match = re.search(r"\d+", result)
            if match:
                score = int(match.group())
                score = max(0, min(100, score))  # 确保在 0-100 范围内
                logger.info(f"[OPENING] LLM 话术匹配评分: {score}")
                return score, score >= threshold
            else:
                logger.warning(f"[OPENING] LLM 返回无效结果: {result}，回退到规则匹配")
        except Exception as e:
            logger.warning(f"[OPENING] LLM 调用失败: {e}，回退到规则匹配")

    # 回退方案：使用规则匹配（基于关键词覆盖率）
    logger.info("[OPENING] 使用规则匹配（无 LLM 或 LLM 失败）")

    # 提取脚本中的关键词/短语（回退规则用）
    key_phrases = [
        "challenge",
        "problem",
        "solution",
        "help",
        "improve",
        "process",
        "current",
        "consider",
        "benefit",
        "decision",
        "15 to 30 minute",
        "working call",
        "meeting",
        "schedule",
    ]

    dialer_lower = dialer_full.lower()
    hits = sum(1 for phrase in key_phrases if phrase.lower() in dialer_lower)
    rule_score = int(round(hits / len(key_phrases) * 100))

    logger.info(
        f"[OPENING] 规则匹配评分: {rule_score} (关键词覆盖: {hits}/{len(key_phrases)})"
    )
    return rule_score, rule_score >= threshold


async def _compute_question_usage(
    dialer_texts: List[str], questions: List[str]
) -> Tuple[List[int], int]:
    """
    先用规则判定；有 LLM 时，用 LLM 对每个问题做一次“是否已问”判定，阈值 50。
    """
    if not questions:
        return [], 0

    # 规则初判
    asked = set()
    for idx, q in enumerate(questions):
        for t in dialer_texts:
            if _similarity(t, q) >= QUESTION_SIM_THRESHOLD:
                asked.add(idx)
                break

    # LLM 精修：覆盖或补充 asked
    if _get_llm():
        dialer_full = " ".join(dialer_texts)
        for idx, q in enumerate(questions):
            prompt = (
                "Determine if the speaker (dialer) has already ASKED the following question "
                "in their utterances. Answer with a number 0-100 (confidence of asked). "
                "Only output the number.\n\n"
                f"Dialer utterances:\n{dialer_full}\n\n"
                f"Target question:\n{q}"
            )
            score = await _ask_score(prompt)
            if score >= 50:  # LLM 认为问过
                asked.add(idx)

    usage_rate = int(round(len(asked) / len(questions) * 100))
    return list(asked), usage_rate


def _compute_answer_match(
    receiver_text: str, qas: List[Dict[str, Any]], asked_indices: List[int]
) -> Optional[int]:
    # 仅统计有答案的问题
    answered_total = 0
    answered_hit = 0
    for idx in asked_indices:
        qa = qas[idx]
        answers = qa.get("answers") or []
        if not answers:
            continue
        answered_total += 1
        best = _best_similarity(receiver_text, answers)
        if best >= QUESTION_SIM_THRESHOLD:
            answered_hit += 1
    if answered_total == 0:
        return None
    return int(round(answered_hit / answered_total * 100))


async def _process_record(
    rec: CallRecord, qas: List[Dict[str, Any]], thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """
    Process a single call record with incremental matching.
    Only calculates scores that are NULL/empty - skips already computed ones.
    """

    # Helper to safely convert database values to int (may be str from DB)
    def safe_int(val):
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    # Convert existing values from DB (may be strings or None)
    existing_opening = safe_int(rec.existing_opening_match)
    existing_question = safe_int(rec.existing_question_match)
    existing_answer = safe_int(rec.existing_answer_match)

    # Check which scores need to be calculated (None means needs calculation)
    # Refined logic: If call_opening is empty, we don't care if opening_match is None - we can't calculate it anyway.
    if not rec.call_opening:
        need_opening = False
        logger.debug(
            f"[STEP2] ID={rec.id} has no call_opening. Skipping opening match but will proceed with question/answer match if needed."
        )
    else:
        need_opening = existing_opening is None

    need_question = existing_question is None
    need_answer = existing_answer is None

    # Skip if all scores already exist
    if not need_opening and not need_question and not need_answer:
        return {
            "callId": rec.id,
            "contact_company": rec.contact_company,
            "processed": False,
            "reason": "all_scores_exist",
            "skipped_incremental": True,
        }

    parsed, segments, _ = _parse_transcription(rec.transcription)
    if not segments:
        return {
            "callId": rec.id,
            "contact_company": rec.contact_company,
            "processed": False,
            "reason": "invalid_segments",
        }

    dialer, receiver = _identify_speakers(segments)
    dialer_texts = [s.get("text", "") for s in segments if s.get("speaker") == dialer]
    receiver_texts = [
        s.get("text", "") for s in segments if s.get("speaker") == receiver
    ]
    receiver_full = " ".join(receiver_texts)

    questions = [qa["question"] for qa in qas]

    # Initialize with existing values or None (using values converted earlier)
    opening_rate = existing_opening
    opening_match = (
        existing_opening is not None and existing_opening >= thresholds["opening"]
    )
    question_usage_rate = existing_question
    question_match = (
        existing_question is not None and existing_question >= thresholds["usage"]
    )
    answer_match_rate = existing_answer
    answer_match = (
        existing_answer is not None and existing_answer >= thresholds["answer"]
    )

    # Track which were newly calculated
    calculated = []

    # Opening: Calculate only if needed
    # 新逻辑：匹配 Dialer 全部发言与完整话术脚本（call_opening + FOLLOW_UP_SCRIPT）
    if need_opening:
        opening_rate, opening_match = await _compute_opening(
            dialer_texts,
            rec.call_opening,  # 传入该公司的 call_opening
            thresholds["opening"],
        )
        calculated.append("opening")

    # Question: Calculate only if needed
    asked_indices = []
    if need_question:
        asked_indices, question_usage_rate = await _compute_question_usage(
            dialer_texts, questions
        )
        question_match = question_usage_rate >= thresholds["usage"]
        calculated.append("question")

    # Answer: Calculate only if needed
    if need_answer:
        # If question was already computed, we need asked_indices for answer
        if not need_question:
            # Recalculate asked_indices for answer computation
            asked_indices, _ = await _compute_question_usage(dialer_texts, questions)

        answer_match_rate = _compute_answer_match(receiver_full, qas, asked_indices)
        if answer_match_rate is None:
            # Fix: If no questions matched or no answers configured, default to 0 as per user request to ensure write-back
            answer_match_rate = 0

        answer_match = (
            answer_match_rate is not None and answer_match_rate >= thresholds["answer"]
        )
        if _get_llm() and asked_indices:
            hit = 0
            total = 0
            for idx in asked_indices:
                qa = qas[idx]
                answers = qa.get("answers") or []
                if not answers:
                    continue
                total += 1
                prompt = (
                    "Judge if the receiver's response matches ANY of the given standard answers "
                    "(semantic match, paraphrase acceptable). Output 0-100.\n\n"
                    f"Question:\n{qa['question']}\n\n"
                    f"Receiver text:\n{receiver_full}\n\n"
                    f"Standard answers:\n" + "\n".join(answers)
                )
                score = await _ask_score(prompt)
                if score >= thresholds["answer"]:
                    hit += 1
            if total > 0:
                answer_match_rate = int(round(hit / total * 100))
                answer_match = answer_match_rate >= thresholds["answer"]
        calculated.append("answer")

    # Fallback: Ensure answer_match_rate is not None if we claim to have calculated it
    if "answer" in calculated and answer_match_rate is None:
        answer_match_rate = 0
        answer_match = False

    # DEBUG LOG START
    logger.debug(
        f"[DEBUG_TRACE] ID={rec.id} | need_answer={need_answer} | answer_match_rate={answer_match_rate}"
    )
    # DEBUG LOG END

    return {
        "callId": rec.id,
        "contact_company": rec.contact_company,  # For logging
        "dialer": dialer,
        "receiver": receiver,
        "opening_match": opening_match,
        "opening_match_rate": opening_rate,
        "question_match": question_match,
        "question_usage_rate": question_usage_rate,
        "answer_match": answer_match,
        "answer_match_rate": answer_match_rate,
        "processed": True,
        "calculated_fields": calculated,  # Track which fields were newly calculated
        "need_opening": need_opening,
        "need_question": need_question,
        "need_answer": need_answer,
    }


def _write_back(results: List[Dict[str, Any]]):
    """Write back results to database, only updating newly calculated fields."""
    conn = get_conn()
    total_writes = 0
    successful_writes = 0
    failed_writes = 0

    try:
        with conn.cursor() as cur:
            for res in results:
                call_id = res.get("callId")
                company = res.get("contact_company", "N/A")

                if not res.get("processed"):
                    reason = res.get("reason", "unknown")
                    skipped_inc = res.get("skipped_incremental", False)
                    if skipped_inc:
                        logger.debug(
                            f"[STEP2] ID={call_id} | Company={company} | SKIPPED | All scores already exist"
                        )
                    else:
                        logger.debug(
                            f"[STEP2] ID={call_id} | Company={company} | SKIPPED | Reason: {reason}"
                        )
                    continue

                # Get which fields were calculated
                calculated = res.get("calculated_fields", [])
                if not calculated:
                    logger.debug(
                        f"[STEP2] ID={call_id} | Company={company} | SKIPPED | No new fields to write"
                    )
                    continue

                # Build dynamic UPDATE statement - only include non-None values
                set_clauses = []
                values = []
                write_fields = []
                skip_reasons = []

                if "opening" in calculated:
                    opening_val = res.get("opening_match_rate")
                    if opening_val is not None:
                        set_clauses.append("opening_match=%s")
                        values.append(opening_val)
                        write_fields.append(f"opening={opening_val}")
                    else:
                        skip_reasons.append("opening(无call_opening)")

                if "question" in calculated:
                    question_val = res.get("question_usage_rate")
                    if question_val is not None:
                        set_clauses.append("question_match=%s")
                        values.append(question_val)
                        write_fields.append(f"question={question_val}")
                    else:
                        skip_reasons.append("question(无问题)")

                if "answer" in calculated:
                    answer_val = res.get("answer_match_rate")
                    if answer_val is not None:
                        set_clauses.append("answer_match=%s")
                        values.append(answer_val)
                        write_fields.append(f"answer={answer_val}")
                    else:
                        skip_reasons.append("answer(无问题匹配)")

                if not set_clauses:
                    # All calculated values are None - log once with clear reason
                    logger.debug(
                        f"[STEP2] ID={call_id} | Company={company} | 无可写数据 | "
                        f"跳过原因: {', '.join(skip_reasons)}"
                    )
                    continue

                values.append(call_id)
                sql = f"UPDATE sdr_cdr_inout_data SET {', '.join(set_clauses)} WHERE id=%s"

                total_writes += 1

                try:
                    cur.execute(sql, tuple(values))
                    rows_affected = cur.rowcount

                    if rows_affected > 0:
                        successful_writes += 1
                        # 构建详细的字段值信息
                        field_value_pairs = []
                        if (
                            "opening" in calculated
                            and res.get("opening_match_rate") is not None
                        ):
                            field_value_pairs.append(
                                f"opening_match={res.get('opening_match_rate')}"
                            )
                        if (
                            "question" in calculated
                            and res.get("question_usage_rate") is not None
                        ):
                            field_value_pairs.append(
                                f"question_match={res.get('question_usage_rate')}"
                            )
                        if (
                            "answer" in calculated
                            and res.get("answer_match_rate") is not None
                        ):
                            field_value_pairs.append(
                                f"answer_match={res.get('answer_match_rate')}"
                            )

                        logger.info(
                            f"[STEP2] ID={call_id} | Company={company} | ✓ 写入成功 | "
                            f"写入字段: {', '.join(field_value_pairs)} | 影响行数: {rows_affected}"
                        )
                    else:
                        failed_writes += 1
                        logger.warning(
                            f"[STEP2] ID={call_id} | Company={company} | ⚠ 写入无效 | "
                            f"记录可能不存在 | 尝试写入: {', '.join(write_fields)}"
                        )
                except Exception as e:
                    failed_writes += 1
                    logger.error(
                        f"[STEP2] ID={call_id} | Company={company} | ✗ 写入失败 | "
                        f"错误: {str(e)} | 尝试写入: {', '.join(write_fields)}"
                    )

            conn.commit()
            logger.info(
                f"[STEP2] === WRITE SUMMARY === | "
                f"Total attempts: {total_writes} | "
                f"Successful: {successful_writes} | "
                f"Failed/No effect: {failed_writes}"
            )
    except Exception as e:
        logger.error(f"[STEP2] Database commit failed: {str(e)}")
        raise
    finally:
        conn.close()


async def main():
    logger.info("=== 分析步骤 2-6 开始 ===")
    qas = _load_questions_answers()
    records = await asyncio.to_thread(_fetch_records)
    logger.info(f"读取到记录数: {len(records)} (date >= {DATE_CUTOFF})")

    thresholds = {
        "usage": (
            float(config.__dict__.get("USAGE_THRESHOLD", 50))
            if hasattr(config, "USAGE_THRESHOLD")
            else 50.0
        ),
        "opening": (
            float(config.__dict__.get("OPENING_THRESHOLD", 70))
            if hasattr(config, "OPENING_THRESHOLD")
            else 70.0
        ),
        "answer": (
            float(config.__dict__.get("ANSWER_THRESHOLD", 70))
            if hasattr(config, "ANSWER_THRESHOLD")
            else 70.0
        ),
    }

    results: List[Dict[str, Any]] = []
    skipped_not_real = 0

    for rec in records:
        logger.info(
            f"[STEP2] 检查记录 id={rec.id} date={rec.date} talking_time={rec.talking_time}s real_person={rec.real_person}"
        )

        # 检查 real_person 字段 (可能是 0, 1, True, False, None)
        # 0 或 False 都表示非真人
        if rec.real_person is not None and (
            rec.real_person == 0 or rec.real_person is False
        ):
            # real_person 明确为 0 或 False，跳过此记录
            logger.info(
                f"[STEP2] 跳过记录 id={rec.id} | Company={rec.contact_company} | real_person={rec.real_person} (非真人)"
            )
            skipped_not_real += 1
            continue

        if rec.real_person is None:
            # real_person 为空，需要通过 LLM 检测
            logger.info(f"[STEP2] 记录 id={rec.id} real_person 为空，开始 LLM 检测...")

            # 解析 transcription 获取 receiver 文本
            parsed, segments, _ = _parse_transcription(rec.transcription)
            if not segments:
                logger.warning(f"[STEP2] 记录 id={rec.id} 无法解析 segments，跳过检测")
                skipped_not_real += 1
                continue

            dialer, receiver = _identify_speakers(segments)
            receiver_texts = [
                s.get("text", "") for s in segments if s.get("speaker") == receiver
            ]

            # 使用 LLM 检测是否为真人
            dialer_texts_for_detect = [
                s.get("text", "") for s in segments if s.get("speaker") == dialer
            ]
            is_real = await _detect_real_person(
                receiver_texts, dialer_texts_for_detect, segments
            )

            # 更新数据库
            await asyncio.to_thread(_update_real_person, rec.id, is_real)

            if not is_real:
                logger.info(
                    f"[STEP2] 跳过记录 id={rec.id} | Company={rec.contact_company} | LLM 检测为非真人"
                )
                skipped_not_real += 1
                continue

            logger.info(f"[STEP2] 记录 id={rec.id} LLM 检测为真人，继续处理")

        # real_person 为 True 或刚检测为真人，继续正常处理
        logger.info(f"[STEP2] 处理记录 id={rec.id} date={rec.date}")
        res = await _process_record(rec, qas, thresholds)
        results.append(res)

    await asyncio.to_thread(_write_back, results)

    valid = sum(1 for r in results if r.get("processed"))
    skipped = len(results) - valid
    logger.info(
        f"✓ 分析完成 | 有效: {valid} | 跳过(解析失败): {skipped} | 跳过(非真人): {skipped_not_real}"
    )


if __name__ == "__main__":
    asyncio.run(main())

import json
from typing import List, Dict, Any
import numpy as np
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.core.logger import logger
from langchain_core.messages import HumanMessage


def eval_faithfulness(answer: str, contexts: List[str]) -> float:
    if not answer or not contexts:
        return 0.0
    combined_ctx = "\n".join(contexts)

    decom_prompt = f"""请将以下回答拆分为若干个独立的原子陈述（每个陈述只表达一个事实）。
以 JSON 数组格式返回，如：["陈述1", "陈述2", ...]

回答：{answer}"""
    llm = get_llm_client(json_mode=True)
    resp = llm.invoke([HumanMessage(content=decom_prompt)])
    try:
        claims = json.loads(resp.content)
    except json.JSONDecodeError:
        logger.warning(f"[RAG评估] faithfulness 拆分解析失败，原始响应: {resp.content[:100]}")
        return 0.0
    if not claims:
        return 1.0

    supported = 0
    for claim in claims:
        check_prompt = f"""请判断以下「陈述」是否可以被「上下文」中的信息直接支持。
以 JSON 格式返回，如：{{"supported": true}} 或 {{"supported": false}}

陈述：{claim}

上下文：{combined_ctx}"""
        resp = llm.invoke([HumanMessage(content=check_prompt)])
        try:
            result = json.loads(resp.content)
            if result.get("supported"):
                supported += 1
        except json.JSONDecodeError:
            if resp.content.strip().lower() == "true":
                supported += 1

    score = supported / len(claims)
    logger.info(f"[RAG评估] faithfulness={score:.4f} ({supported}/{len(claims)})")
    return score


def eval_answer_relevancy(question: str, answer: str) -> float:
    if not question or not answer:
        return 0.0

    gen_prompt = f"""请根据以下答案，反向生成 3 个可能对应的问题（只输出 JSON 数组）：

答案：{answer}"""
    llm = get_llm_client(json_mode=True)
    resp = llm.invoke([HumanMessage(content=gen_prompt)])
    try:
        hypo_questions = json.loads(resp.content)
    except json.JSONDecodeError:
        logger.warning(f"[RAG评估] answer_relevancy 生成问题解析失败")
        return 0.0
    if not hypo_questions:
        return 0.0

    all_texts = [question] + hypo_questions
    embeds = generate_embeddings(all_texts)
    orig_vec = embeds["dense"][0]
    scores = []
    for h_vec in embeds["dense"][1:]:
        cos_sim = float(np.dot(orig_vec, h_vec) / (np.linalg.norm(orig_vec) * np.linalg.norm(h_vec) + 1e-9))
        scores.append(cos_sim)
    score = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"[RAG评估] answer_relevancy={score:.4f}")
    return score


def eval_context_precision(question: str, contexts: List[str]) -> float:
    if not question or not contexts:
        return 0.0

    llm = get_llm_client()
    relevant_at_k = []
    for i, ctx in enumerate(contexts, 1):
        judge_prompt = f"""问题：{question}
文档：{ctx[:500]}

该文档是否与问题相关？只回答 true 或 false。"""
        resp = llm.invoke([HumanMessage(content=judge_prompt)])
        relevant_at_k.append(resp.content.strip().lower() == "true")

    relevant_count = sum(relevant_at_k)
    if relevant_count == 0:
        return 0.0

    precisions = []
    for i, is_rel in enumerate(relevant_at_k, 1):
        if is_rel:
            precisions.append(sum(relevant_at_k[:i]) / i)
    score = sum(precisions) / relevant_count
    logger.info(f"[RAG评估] context_precision={score:.4f} (相关{relevant_count}/{len(contexts)})")
    return score


def full_evaluation(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    try:
        return {
            "faithfulness": eval_faithfulness(answer, contexts),
            "answer_relevancy": eval_answer_relevancy(question, answer),
            "context_precision": eval_context_precision(question, contexts),
        }
    except Exception as e:
        logger.error(f"[RAG评估] 全量评估异常: {e}", exc_info=True)
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}

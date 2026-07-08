from langchain_core.tools import tool
from app.core.logger import logger
from app.query_process.agent.nodes.node_item_name_confirm import step_3_extract_info


@tool
def identify_item_names(query: str, history: list[dict] = None) -> dict:
    """识别用户问题中涉及的商品名称，并重写问题使其独立完整。query: 用户原始问题; history: 历史对话列表（可选）"""
    logger.info(f"[Agent Tool] identify_item_names: query={query}")

    result = step_3_extract_info(query, history or [])
    item_names = result.get("item_names", [])
    rewritten = result.get("rewritten_query", query)
    logger.info(f"[Agent Tool] identified items: {item_names}, rewritten: {rewritten[:50]}")
    return {"item_names": item_names, "rewritten_query": rewritten}

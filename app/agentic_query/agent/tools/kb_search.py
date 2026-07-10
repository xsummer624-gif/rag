from langchain_core.tools import tool
from app.core.logger import logger
from app.query_process.agent.nodes.node_search_embedding import search_embedding

MAX_CHARS = 300


def _truncate(content: str) -> str:
    return content[:MAX_CHARS] + "..." if len(content) > MAX_CHARS else content


@tool
def search_knowledge_base(query: str, item_names: list[str] = None) -> list[dict]:
    """在本地知识库中搜索与问题相关的技术文档片段（向量检索）。query: 搜索问题; item_names: 限定商品名列表（可选）"""
    if isinstance(item_names, str):
        item_names = [item_names]

    logger.info(f"[Agent Tool] search_knowledge_base: query={query}, item_names={item_names}")

    results = search_embedding(query=query, item_names=item_names)
    for r in results:
        entity = r.get("entity", {})
        if "content" in entity:
            entity["_full_content"] = entity.get("content", "")
            entity["content"] = _truncate(entity.get("content", ""))
    logger.info(f"[Agent Tool] KB search found {len(results)} chunks")
    return results

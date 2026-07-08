import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from app.core.logger import logger
from app.conf.lm_config import lm_config
from app.utils.task_utils import (
    update_task_status, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
    set_task_result,
)
from app.utils.sse_utils import push_to_session, SSEEvent
from app.clients.mongo_history_utils import save_chat_message, get_recent_messages
from app.evaluation.ragas_metrics import full_evaluation
from app.query_process.agent.nodes.node_item_name_confirm import (
    step_3_extract_info,
    step_4_vectorize_and_query,
    step_5_align_item_names,
)
from app.agentic_query.agent.tools.kb_search import search_knowledge_base
from app.agentic_query.agent.tools.hyde_search import search_knowledge_base_enhanced
from app.agentic_query.agent.tools.web_search import search_web

tools = [search_knowledge_base, search_knowledge_base_enhanced, search_web]

BASE_PROMPT = """你是掌柜智库的智能客服助手。你必须先搜索再回答，不得跳过工具直接回答。

## 可用工具
- search_knowledge_base(query, item_names): 普通向量搜索知识库
- search_knowledge_base_enhanced(query, item_names): HyDE 增强搜索（适合模糊查询）
- search_web(query): 联网搜索实时信息

## 搜索规则
- 至少调用 1 次知识库搜索工具（kb 或 hyde）。
- 搜索工具**总共最多调用 3 次**，三种工具各调用 1 次即已充分。
- 工具返回空结果时换另一个工具搜索。
- 所有工具均返回空时，基于自身知识回答，注明"数据库中未找到相关信息"。

## 终止条件
- 搜索完成后**立即**综合所有结果给出最终回答，不得重复搜索。
- 禁止连续调用同一个工具。

## 回答要求
- 引用来源
- 需要图片时追加【图片】区块，每行一个URL
- 简洁有条理"""


def build_system_prompt(db_item_names: list, rewritten_query: str) -> str:
    ctx = ""
    if db_item_names:
        ctx += f"\n\n## 当前主题识别结果（已确认的商品名）\n商品名: {', '.join(db_item_names)}"
    if rewritten_query:
        ctx += f"\n改写后的问题: {rewritten_query}"
    if ctx:
        ctx += "\n\n搜索工具中的 item_names 参数请使用上述已确认的商品名。"
    return BASE_PROMPT + ctx


def _run_topic_identification(user_query: str, history: list) -> dict:
    """强制运行主题识别：step3 LLM提取 → step4 向量对齐 → step5 确认"""
    result = {"db_item_names": [], "rewritten_query": user_query, "error": None}
    try:
        extracted = step_3_extract_info(user_query, history)
        rewritten_query = extracted.get("rewritten_query", user_query)
        extracted_names = extracted.get("item_names", [])
        result["rewritten_query"] = rewritten_query

        if extracted_names:
            vector_results = step_4_vectorize_and_query(extracted_names)
            align_result = step_5_align_item_names(vector_results)
            confirmed = align_result.get("confirmed_item_names", []) or align_result.get("options", [])
            if confirmed:
                result["db_item_names"] = confirmed
                logger.info(f"[主题识别] 商品名对齐成功: {confirmed}")
            else:
                logger.info(f"[主题识别] 未找到匹配的数据库商品名，使用原始提取名: {extracted_names}")
                result["db_item_names"] = extracted_names
        else:
            logger.info("[主题识别] 未提取到商品名")
    except Exception as e:
        logger.warning(f"[主题识别] 异常(不影响主流程): {e}")
        result["error"] = str(e)
    return result


def extract_agent_chain(messages) -> list:
    chain = []
    pending = []
    for msg in messages:
        tc = getattr(msg, "tool_calls", None) or []
        for call in tc:
            entry = {"tool": call.get("name", ""), "input": call.get("args", {}), "output": None}
            pending.append(entry)
            chain.append(entry)
        if hasattr(msg, "name") and getattr(msg, "type", "") == "tool":
            for entry in reversed(pending):
                if entry["output"] is None:
                    content = msg.content or ""
                    entry["output"] = content[:500] if len(content) > 500 else content
                    break
    return chain


model = ChatOpenAI(
    model=lm_config.llm_model,
    openai_api_key=lm_config.api_key,
    openai_api_base=lm_config.base_url,
    temperature=lm_config.llm_temperature,
)

agent_executor = create_react_agent(
    model=model,
    tools=tools,
    name="agentic_rag_agent",
)


def run_agentic_graph(state: dict) -> dict:
    session_id = state["session_id"]
    user_query = state["original_query"]
    is_stream = state.get("is_stream", False)
    logger.info(f"[AgenticRAG] 开始处理: session={session_id}, query={user_query}")

    # 加载历史
    try:
        history = get_recent_messages(session_id, limit=10)
    except Exception:
        history = []

    # === [强制] 主题识别预处理 ===
    topic = _run_topic_identification(user_query, history)
    db_item_names = topic["db_item_names"]
    rewritten_query = topic["rewritten_query"]

    # 构造消息：动态 system prompt + 历史 + 当前问题
    system_prompt = build_system_prompt(db_item_names, rewritten_query)
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        role = msg.get("role", "")
        text = msg.get("text", "")
        if role == "user":
            messages.append(HumanMessage(content=text))
        elif role in ("assistant", "bot"):
            messages.append(AIMessage(content=text))
    messages.append(HumanMessage(content=user_query))

    try:
        result = agent_executor.invoke(
            {"messages": messages},
            {"recursion_limit": 10, "configurable": {"recursion_limit": 10}},
        )
        final_msg = result["messages"][-1]
        answer = final_msg.content

        agent_chain = extract_agent_chain(result["messages"])

        # 将主题识别作为第一步加入链
        topic_step = {
            "tool": "topic_identification",
            "input": {"query": user_query, "identified_names": db_item_names},
            "output": json.dumps({"db_item_names": db_item_names, "rewritten_query": rewritten_query}, ensure_ascii=False),
        }
        agent_chain = [topic_step] + agent_chain

        # 提取 tool_results
        tool_results = {}
        for msg in result["messages"]:
            if hasattr(msg, "name") and msg.name in ("search_knowledge_base", "search_knowledge_base_enhanced"):
                try:
                    data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(data, list):
                        tool_results.setdefault("contexts", []).extend(
                            [d.get("entity", d).get("content", "") for d in data if isinstance(d, dict)]
                        )
                except (json.JSONDecodeError, TypeError):
                    pass

        state["answer"] = answer
        state["agent_chain"] = agent_chain
        set_task_result(session_id, "answer", answer)
        set_task_result(session_id, "agent_chain", json.dumps(agent_chain, ensure_ascii=False))
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)

        # 保存历史
        try:
            save_chat_message(session_id=session_id, role="user", text=user_query)
            save_chat_message(session_id=session_id, role="assistant", text=answer)
        except Exception as e:
            logger.warning(f"[AgenticRAG] 保存历史失败: {e}")

        # RAGAS
        rag_scores_final = None
        try:
            contexts = tool_results.get("contexts", [])
            scores = full_evaluation(question=user_query, answer=answer, contexts=contexts)
            scores["session_id"] = session_id
            rag_scores_final = scores
            set_task_result(session_id, "rag_scores", json.dumps(scores, ensure_ascii=False))
            try:
                save_chat_message(session_id=session_id, role="system", text=f"[RAG评估] {json.dumps(scores, ensure_ascii=False)}")
            except Exception:
                pass
            logger.info(f"[AgenticRAG] RAG评估: {scores}")
        except Exception as e:
            logger.warning(f"[AgenticRAG] 评估异常(不影响): {e}")

        # SSE
        if is_stream:
            try:
                push_to_session(session_id, SSEEvent.FINAL, {
                    "answer": answer, "status": "completed", "image_urls": [],
                    "agent_chain": agent_chain, "rag_scores": rag_scores_final,
                })
            except Exception:
                pass

        logger.info(f"[AgenticRAG] 处理完成: session={session_id}")
        return state

    except Exception as e:
        logger.error(f"[AgenticRAG] 流程异常: {e}", exc_info=True)
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        state["answer"] = f"处理失败: {e}"
        return state

import json
import uuid
import uvicorn
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.core.logger import logger
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.agentic_query.agent.state import create_default_state
from app.agentic_query.agent.agentic_graph import run_agentic_graph

try:
    from app.clients.mongo_history_utils import *
    _mongo_available = True
except Exception:
    logger.warning("MongoDB 不可用，历史对话功能将跳过")
    _mongo_available = False

app = FastAPI(title="Agentic RAG Query Service", description="掌柜智库智能Agent查询服务！")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "agentic"}


@app.get("/chat.html")
async def chat_html():
    chat_html_path = PROJECT_ROOT / 'app' / 'agentic_query' / 'page' / 'chat.html'
    if not chat_html_path.exists():
        chat_html_path = PROJECT_ROOT / 'app' / 'query_process' / 'page' / 'chat.html'
    return FileResponse(chat_html_path, headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })


class QueryRequest(BaseModel):
    query: str = Field(..., title="查询内容")
    session_id: str = Field(None, title="会话id")
    is_stream: bool = Field(False, title="是否流式返回结果")


async def run_agentic(session_id: str, user_query: str, is_stream: bool = True):
    state = create_default_state(
        original_query=user_query,
        session_id=session_id,
        is_stream=is_stream,
    )
    try:
        await run_agentic_graph(state)
    except Exception as e:
        print(f"[Agentic] 流程异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})


@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream

    if is_stream:
        create_sse_queue(session_id)

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    logger.info(f"[Agentic] 请求: session={session_id}, query={user_query}, stream={is_stream}")

    if is_stream:
        background_tasks.add_task(run_agentic, session_id, user_query, is_stream)
        return {"message": "结果正在处理中...", "session_id": session_id}
    else:
        await run_agentic(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        rag_raw = get_task_result(session_id, "rag_scores", "")
        rag_scores = json.loads(rag_raw) if rag_raw else None
        chain_raw = get_task_result(session_id, "agent_chain", "")
        agent_chain = json.loads(chain_raw) if chain_raw else None
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            "done_list": [],
            "rag_scores": rag_scores,
            "agent_chain": agent_chain,
        }


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)

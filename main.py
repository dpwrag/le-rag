import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from setup import create_graph
from api.schemas import QueryRequest, QueryResponse, SelectedMenu

logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variable for graph
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize graph on startup and clean up on shutdown."""
    global graph

    logger.info("Initializing LE-RAG application...")
    graph = create_graph()
    logger.info("Agent graph created successfully")

    yield

    logger.info("Shutting down application...")


app = FastAPI(
    title="LE-RAG API",
    description="API for querying the Restaurant Menu using Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "graph_ready": graph is not None,
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query through the agent graph.

    Args:
        request: QueryRequest containing the user query

    Returns:
        QueryResponse with the processed results
    """
    if not graph:
        logger.error("Graph not initialized")
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Graph not initialized.",
        )

    try:
        logger.info(f"Processing query: {request.query}")

        final_state = graph.invoke(
            {"messages": [HumanMessage(content=request.query)]},
            config={"configurable": {"thread_id": "session-1"}},
        )

        messages = final_state.get("messages", [])

        serialized_messages = []
        for msg in messages:
            if hasattr(msg, "content"):
                serialized_messages.append(
                    {
                        "role": msg.__class__.__name__,
                        "content": msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content),
                    }
                )
            else:
                serialized_messages.append({"role": "unknown", "content": str(msg)})

        logger.info(
            f"Query processed successfully. Generated {len(serialized_messages)} messages"
        )

        selected_menu = []
        selected_menu_list = final_state.get("selected_menu_list")
        if selected_menu_list:
            if hasattr(selected_menu_list, "menu_list"):
                for menu_item in selected_menu_list.menu_list:
                    selected_menu.append(
                        SelectedMenu(
                            name=menu_item.name,
                            description=menu_item.description,
                            score=menu_item.score,
                            justification=menu_item.justification,
                        )
                    )

        return QueryResponse(
            query=request.query,
            messages=serialized_messages,
            selected_menu=selected_menu,
            thread_id="session-1",
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

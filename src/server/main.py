from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from open_deep_research.multi_agent import graph

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class UserMessage(BaseModel):
    content: str

@app.get("/")
async def healthcheck():
    logger.debug("Health check endpoint called")
    return {"status": "healthy"}

@app.post("/research")
async def run_langgraph_endpoint(user_message: UserMessage):
    try:
        # Initialize and run the LangGraph with the user message
        initial_state = {"messages": [{"role": "user", "content": user_message.content}]}
        logger.info(f"Invoking deep_research with initial state: {initial_state}")
        graph_state = await graph.ainvoke(initial_state)
        logger.info("Research completed successfully")
        return JSONResponse(content=graph_state)
    except Exception as e:
        logger.error(f"Error occurred performing research: {e}")
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from open_deep_research.multi_agent import graph

app = FastAPI()

class UserMessage(BaseModel):
    content: str

@app.get("/")
async def healthcheck():
    return {"status": "healthy"}

@app.post("/research")
async def run_langgraph_endpoint(user_message: UserMessage):
    try:
        # Initialize and run the LangGraph with the user message
        initial_state = {"messages": [{"role": "user", "content": user_message.content}]}
        print(f"Invoking deep_research with initial state: {initial_state}")
        graph_state = await graph.ainvoke(initial_state)
        return JSONResponse(content=graph_state)
    except Exception as e:
        print(f"Error occurred performing research:", e)
        raise HTTPException(status_code=500, detail=str(e))
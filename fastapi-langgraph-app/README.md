# FastAPI LangGraph Application

This project is a FastAPI application that integrates with LangGraph to manage and execute workflows directly from Python. It provides a simple API for initiating LangGraph runs and checking the health of the server.

## Project Structure

```
fastapi-langgraph-app
├── src
│   ├── main.py              # Entry point of the FastAPI application
│   ├── langgraph_runner.py   # Logic to initiate LangGraph runs
│   └── models
│       └── request.py       # Request models for FastAPI
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fastapi-langgraph-app
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the FastAPI application:**
   ```bash
   uvicorn src.main:app --reload
   ```

2. **Healthcheck Endpoint:**
   - **GET** `/healthcheck`
   - Returns a simple health status of the server.

3. **Start LangGraph Run:**
   - **POST** `/run`
   - Request Body:
     ```json
     {
       "user_message": "Your message content here"
     }
     ```
   - This endpoint initiates a LangGraph run with the provided user message and waits for its completion before responding with the entire graph state.

## Example Request

To start a LangGraph run, you can use `curl` or any API client:

```bash
curl -X POST "http://localhost:8000/run" -H "Content-Type: application/json" -d '{"user_message": "What is the capital of France?"}'
```

## Notes

- Ensure that all dependencies are installed correctly.
- The application is designed to run locally and can be extended for production use.
- For more information on LangGraph, refer to its official documentation.
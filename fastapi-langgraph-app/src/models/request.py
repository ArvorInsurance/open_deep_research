from pydantic import BaseModel

class UserMessageRequest(BaseModel):
    message: str

class HealthCheckResponse(BaseModel):
    status: str
    message: str
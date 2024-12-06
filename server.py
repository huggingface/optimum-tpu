from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    inputs: str

@app.post("/generate")
async def generate(request: GenerateRequest):
    return {
        "generated_text": "Hello World!",
        "request_received": request.dict()
    }

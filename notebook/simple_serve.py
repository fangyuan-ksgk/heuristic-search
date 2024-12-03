# Simple Server for SGLang 
import sglang as sgl
from fastapi import FastAPI, HTTPException
import uvicorn

# Global engine variable
engine = None

# Create FastAPI app
app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/generate")
async def generate(prompts: list[str]):
    try:
        sampling_params = {"temperature": 0.8, "top_p": 0.95}
        outputs = await engine.async_generate(prompts, sampling_params)
        return {"results": [output["text"] for output in outputs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_server():
    global engine
    engine = sgl.Engine(model_path="unsloth/Llama-3.2-11B-Vision-Instruct")
    uvicorn.run(app, host="0.0.0.0", port=30000, single_process=True)

if __name__ == "__main__":
    run_server()
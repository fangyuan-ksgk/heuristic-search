import os
import modal
import warnings

GPU_TYPE = os.environ.get("GPU_TYPE", "a10g")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"
SGL_LOG_LEVEL = "error"
MINUTES = 60

# Model configuration
MODEL_PATH = "unsloth/Llama-3.2-11B-Vision-Instruct"
MODEL_REVISION = None
TOKENIZER_PATH = "unsloth/Llama-3.2-11B-Vision-Instruct"
MODEL_CHAT_TEMPLATE = "llama_3_vision"
PORT_NO = 3333

def download_model_to_image():
    import transformers
    from huggingface_hub import snapshot_download
    snapshot_download(
        MODEL_PATH,
        revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],
    )
    transformers.utils.move_cache()

# Simplify the image definition
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sglang>=0.3.6",
        "transformers>=4.37.2",
        "numpy",
        "fastapi>=0.109.0",
        "pydantic>=2.0",
        "starlette>=0.36.0",
        "openai>=1.40.8",
        "orjson",
        "torch",
        "accelerate",
        "uvicorn",
    )
    .run_function(download_model_to_image)
)

app = modal.App("sgl-vlm-service")

@app.cls(
    gpu=GPU_CONFIG,
    image=image,
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("ksgk-secret")],
    container_idle_timeout=20 * MINUTES,
)
class Model:
    def __init__(self):
        self.client = None
        self.server_process = None

    @modal.enter()
    def start_runtime(self):
        import subprocess
        import time
        
        # Start the server using subprocess instead of the old utility function
        self.server_process = subprocess.Popen([
            "python3", "-m", "sglang.launch_server",
            "--model-path", MODEL_PATH,
            "--port", str(PORT_NO),
            "--chat-template", MODEL_CHAT_TEMPLATE
        ])
        
        # Wait for server to start
        time.sleep(10)  # Give the server some time to start
        
        from openai import OpenAI
        self.client = OpenAI(
            base_url=f"http://localhost:{PORT_NO}/v1", 
            api_key="dummy-key"
        )

    @modal.method()
    def process_chat(self, messages):
        response = self.client.chat.completions.create(
            model=MODEL_PATH,
            messages=messages,
            max_tokens=300,
        )
        return response

    @modal.exit()
    def shutdown_runtime(self):
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()

@app.local_entrypoint()
def main():
    model = Model()
    
    # Test message
    test_messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                }
            }
        ]
    }]
    
    response = model.process_chat.remote(test_messages)
    print("Test response:", response.choices[0].message.content)

# Filter warnings
warnings.filterwarnings(
    "ignore",
    message="It seems this process is not running within a terminal.",
    category=UserWarning,
)


from vllm import LLM, SamplingParams
from PIL import Image
import requests
from io import BytesIO
from typing import List, NamedTuple, Optional

class ModelRequestData(NamedTuple):
    llm: LLM
    prompt: str
    stop_token_ids: Optional[List[str]]
    image_data: List[Image.Image]
    chat_template: Optional[str]

def load_mllama(question, image_urls: List[str]) -> ModelRequestData:
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    prompt = f"<|image|><|image|><|begin_of_text|>{question}"
    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=None,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=None,
    )

def fetch_image(url: str) -> Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def run_llama_inference(question: str, image_urls: list[str]):
    # Get the model request data using the existing load_mllama function
    req_data = load_mllama(question, image_urls)
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic output
        max_tokens=512,   # Adjust based on your needs
        stop_token_ids=None
    )

    # Run inference
    outputs = req_data.llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {
                "image": req_data.image_data
            },
        },
        sampling_params=sampling_params
    )

    # Print the generated text
    for output in outputs:
        print(output.outputs[0].text)

# Example usage
if __name__ == "__main__":
    # Example question and images
    question = "What do you see in these images?"
    image_urls = [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg"
    ]
    
    run_llama_inference(question, image_urls)

        
        

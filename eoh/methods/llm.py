from typing import List, Optional, Union
from transformers import AutoTokenizer
from openai import OpenAI 
from os import getenv
import torch
import random 
import anthropic
import os

oai_client = OpenAI()
claude_client = anthropic.Anthropic()

class OpenRouterModel:
    BASEURL = "https://openrouter.ai/api/v1"
    MODELS = [
        "anthropic/claude-3.5-sonnet",
        "qwen/qwen-110b-chat", 
        "mistralai/mistral-large", 
        "meta-llama/llama-3-70b-instruct:nitro", 
        "mistralai/mistral-nemo", 
        "01-ai/yi-large",
        "cohere/command-r-plus", 
        "openai/gpt-4o-mini",
        "google/gemma-2-27b-it", 
        "microsoft/wizardlm-2-8x22b",
        "deepseek/deepseek-chat"
    ]
    KEY_ENV_VAR = "OPENROUTER_API_KEY"
    MAX_TOKENS = 400
    def __init__(self):
        self.client = OpenAI(
            base_url=self.BASEURL,
            api_key=getenv(self.KEY_ENV_VAR),
        )
    def get_completion(self, system_prompt: str, prompt: str, idx: Optional[int] = None) -> str:
        if idx is None:
            idx = random.randint(0, len(self.MODELS) - 1)
        completion = self.client.chat.completions.create(
            model = self.MODELS[idx],
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        print(f"{self.MODELS[idx]} response: {completion.choices[0].message.content}")
        return completion.choices[0].message.content
    
    
def get_helper_response(prompt, rand=True):
    if rand:
        helper_model_idx = random.randint(0, len(OpenRouterModel.MODELS) - 1)
    else:
        helper_model_idx = 0
    helper_model = OpenRouterModel()
    response = helper_model.get_completion(prompt, helper_model_idx)
    return response


def get_openai_response(input: Union[str, list], model_name="gpt-4o"):
    
    if isinstance(input, str):
        msg = [{"role": "user", "content": input}]
    elif isinstance(input, list):
        msg = input
    else:
        raise ValueError(f"Invalid input type: {type(input)}")
    
    response = oai_client.chat.completions.create(
        model=model_name,
        messages=msg,
    )
    return response.choices[0].message.content


def get_claude_response(prompt: Union[str, list], img = None, img_type = None, system_prompt = "You are a helpful assistant."):
    """ 
    Claude response with query and image input
    """
    
    if isinstance(prompt, str):
        query = prompt
    else:
        if prompt[0]["role"] == "system":
            system_prompt = prompt[0]["content"]
        query = prompt[1:]
        
    if img is not None:
        text_content = [{"type": "text", "text": query}]
        img_content = [{"type": "image", "source": {"type": "base64", "media_type": img_type, "data": img}}]
    else:
        text_content = query
        img_content = ""
        
    message = claude_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": img_content + text_content,
            }
        ],
        system=system_prompt,
    )
    return message.content[0].text 


try:
    from vllm import LLM, SamplingParams
    class VLLM:
        def __init__(
            self,
            name: str,
            # gpu_ids: List[int] = [0, 1], # Assuming we have 2 GPUs here
            download_dir: Optional[str] = None,
            dtype: str = "auto",
            gpu_memory_utilization: float = 0.85,
            max_model_len: int = 4096,
            merge: bool = False,
            max_new_tokens: int = 2048,
            **kwargs,
        ) -> None:
            self.name: str = name
            if merge:
                os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN" # Use this for merged model
            else:
                # os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" # Use this for baseline model | Gemma2 require this backend for inference
                os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN" # Default to using llama3.1 70B (quantized version, of course)            
            
            available_gpus = list(range(torch.cuda.device_count()))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))
            
            # if len(available_gpus) > 1:
            #     import multiprocessing
            #     multiprocessing.set_start_method('spawn', force=True)

            self.model: LLM = LLM(
                model=self.name,
                tensor_parallel_size=len(available_gpus),
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                download_dir=download_dir,
                max_model_len=max_model_len,
            )
            
            self.params = SamplingParams(**kwargs)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        
        def completions(
            self,
            prompts: List[str],
            use_tqdm: bool = False,
            **kwargs: Union[int, float, str],
        ) -> List[str]:
            formatted_prompts = [self.format_query_prompt(prompt.strip()) for prompt in prompts]
    
            outputs = self.model.generate(formatted_prompts, self.params, use_tqdm=use_tqdm)
            outputs = [output.outputs[0].text for output in outputs]
            return outputs

        def generate(
            self,
            prompts: List[str],
            use_tqdm: bool = False,
            **kwargs: Union[int, float, str],
        ) -> List[str]:
            formatted_prompts = [self.format_query_prompt(prompt.strip()) for prompt in prompts]
            return self.model.generate(formatted_prompts, self.params, use_tqdm=use_tqdm)

        def format_query_prompt(self, prompt: str, completion: str = "####Dummy-Answer") -> str:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
            format_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            query_prompt = format_prompt.split(completion)[0]
            return query_prompt
        
except ImportError:
    class VLLM:
        def __init__(self, *args, **kwargs):
            pass
        def completions(self, *args, **kwargs):
            return get_openai_response(*args, **kwargs)
    
    # Just write a dummy VLLM class for Mac instance here 
    print("Could not load vllm class, check CUDA support and GPU RAM size")
    
    
try:
    import groq
    groq_client = groq.Groq()
    
    def get_groq_response(prompt: str):
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    
except:
    pass
    

# Trainable token on top of a LLM (Qwen-0.5B) in this case
# Temasek Foundation Dataset used here :: Binary + Comment type of data format

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, random
import numpy as np 
import torch.nn as nn
from sklearn.model_selection import train_test_split


def load_hf_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """ 
    Load Huggingface model and tokenizer
    """    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Change to float32 instead of "auto"
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer


def load_tf_data(data_path: str = "../data/processed_data.json", split_ratio: float = 0.8):
    """ 
    Load processed data
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    # train-test split with fixed seed
    train_indices, test_indices = train_test_split(range(len(data['prompt'])), test_size=0.2, random_state=42)
    train_data = {k: [v[i] for i in train_indices] for k, v in data.items()}
    test_data = {k: [v[i] for i in test_indices] for k, v in data.items()}
        
    return train_data, test_data 


# Worth including diversity in response template
RESPONSE_TEMPLATES = [
    "Decision: {label}\n\nComment: {comment}",
    "Comment: {comment}\n\nDecision: {label}",
    "Reasoning: {comment}\n\nDecision: {label}",
    "Decision: {label}\n\nReasoning: {comment}",
    "Comment on the proposal: {comment}\n\nDecision: {label}",
    "Decision: {label}\n\nComment on the proposal: {comment}",
]

SYSTEM_PROMPTS = [
    "You are a grant reviewer for Temasek Foundation. Your task is to review grant applications and provide a decision.",
    "As an expert grant evaluator at Temasek Foundation, assess the following application and provide your decision with supporting comments.",
    "You are an experienced grant assessor for Temasek Foundation. Review this application carefully and determine if it should be approved or rejected.",
    "Working as a Temasek Foundation grant reviewer, evaluate this proposal and provide your professional assessment and final decision."
]

def _form_query(prompt: str) -> str:
    query = prompt
    return query 

def _form_response(label: str, comment: str) -> str:
    template = random.choice(RESPONSE_TEMPLATES)
    response = template.replace("{label}", label).replace("{comment}", comment)
    return response
 
def _form_system_prompt() -> str: 
    system_prompt = random.choice(SYSTEM_PROMPTS)
    return system_prompt 


def format_prompt_instruction_tuned(prompt: str, comment: str, label: str, tokenizer, previous_messages: list = []):
    """ 
    Format single-turn response tuning -- not realistic yet, practical usage should involves multi-turn conversation
    
    Format prompt - response for instruction-tuned LLM 
    """ 
    system_prompt = _form_system_prompt()
    query_str = _form_query(prompt)
    response_str = _form_response(label, comment)    
    
    fillin_response = "####Response####"
    fillin_messages = previous_messages + [{"role": "user", "content": query_str}, 
                                                {"role": "assistant", "content": fillin_response}]
    messages = [{"role": "system", "content": system_prompt}] + fillin_messages
    
    complete_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    query_prompt, suffix_prompt = complete_prompt.split(fillin_response)
    response_prompt = response_str + suffix_prompt
    
    return query_prompt, response_prompt 


# Soft Prompt Wrapper 
class SoftPromptLLM(nn.Module): 
    
    def __init__(self, model, tokenizer, n_learnable_tokens, initialize_from_vocab=False):
        super().__init__()
        self.model = model 
        self.tokenizer = tokenizer 
        self.n_learnable_tokens = n_learnable_tokens
        
        # Get embedding dimension from model
        embedding_size = model.get_input_embeddings().weight.shape[1]
        
        # Get the device from the model
        self.device = next(model.parameters()).device
        
        # Initialize soft prompts
        if initialize_from_vocab:
            # Initialize from random vocabulary tokens
            init_prompt_ids = torch.randint(len(tokenizer), (n_learnable_tokens,))
            init_prompt_ids = init_prompt_ids.to(self.device)
            self.soft_prompts = nn.Parameter(
                model.get_input_embeddings()(init_prompt_ids),
                requires_grad=True
            )
        else:
            # Random initialization
            self.soft_prompts = nn.Parameter(
                torch.randn(n_learnable_tokens, embedding_size, device=self.device),
                requires_grad=True
            )
            
        # freeze other model parameters
        for param in self.model.parameters():
            param.requires_grad = False 
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        
        batch_size = input_ids.shape[0]
        
        # Expand soft prompts for batch size
        soft_prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Concatenate soft prompts with input embeddings
        inputs_embeds = torch.cat([soft_prompts, inputs_embeds], dim=1)
        
        # Adjust attention mask if provided
        if attention_mask is not None:
            prefix_mask = torch.ones(batch_size, self.n_learnable_tokens).to(attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
        # Adjust labels if provided
        if labels is not None: 
            prefix_labels = torch.full((batch_size, self.n_learnable_tokens), -100, dtype=labels.dtype).to(labels.device)
            labels = torch.cat([prefix_labels, labels], dim=1)
        
        # Forward pass through model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
# Trainable token on top of a LLM (Qwen-0.5B) in this case
# Temasek Foundation Dataset used here :: Binary + Comment type of data format

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, random
import numpy as np 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Callable

def load_hf_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """ 
    Load Huggingface model and tokenizer
    """    
    # if on mps device use float32
    if torch.backends.mps.is_available():
        torch_dtype = torch.float32
        device_map = "mps"
    else:
        torch_dtype = torch.float16
        device_map = "cuda"
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,  # Change to float32 instead of "auto"
        device_map=device_map
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


class ClsCommentDataset(Dataset): 
    
    """Dataset for Temasek prompt-based language model training.
    
    Args:
        prompts (list): List of input prompts
        labels (list): List of corresponding labels
        tokenizer: Tokenizer instance
        comments (list, optional): List of comments. Defaults to empty list
        max_length (int, optional): Maximum sequence length. Defaults to 512
    """
    
    def __init__(self, load_data_func: Callable, tokenizer: AutoTokenizer, max_length: int = 512, train: bool = True):
        
        train_data, test_data = load_data_func()
        if train:
            prompts = train_data['prompt']
            labels = train_data['label']
            comments = train_data['comment']
        else:
            prompts = test_data['prompt']
            labels = test_data['label']
            comments = test_data['comment']
            
        if len(prompts) != len(labels):
            raise ValueError("Length of prompts and labels must match")
        self.prompts = prompts
        self.labels = labels
        self.comments = comments
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        comment = self.comments[idx]
        
        # Query and response prompt
        query_prompt, response_prompt = format_prompt_instruction_tuned(prompt, comment, label, self.tokenizer, previous_messages = [])
        
        full_prompt = query_prompt + response_prompt
        
        # Tokenize the full sequence
        tokenized = self.tokenizer(full_prompt, 
                                 return_tensors="pt",
                                 max_length=self.max_length,
                                 truncation=True,
                                 padding="max_length")
        
        # The padding mask (1s for real tokens, 0s for padding) | note for dataset preparation we don't do causal masking, that happen within the model
        attention_mask = tokenized.attention_mask[0]
        
        # Create labels by shifting input_ids right (-100 for query tokens)
        query_length = len(self.tokenizer(query_prompt)['input_ids'])
        labels = tokenized.input_ids.clone()
        labels[0, :query_length] = -100  # Mask out the query portion
        
        # Shift labels right by 1 and add -100 padding
        labels = torch.roll(labels, shifts=1, dims=1)
        labels[0, 0] = -100  # First token has no previous token to predict it
        
        # Set labels for padding tokens to -100
        labels[0, attention_mask == 0] = -100
        
        return {
            'input_ids': tokenized.input_ids[0],
            'attention_mask': attention_mask,
            'labels': labels[0]
        }


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
    
    def forward(self, input_ids, attention_mask=None, labels=None, return_pred_n_label=False, **kwargs):
        
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

        pred_logits = outputs.logits

        if return_pred_n_label:
            return outputs, pred_logits, labels
        elseß:
            return outputs
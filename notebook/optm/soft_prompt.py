# Trainable token on top of a LLM (Qwen-0.5B) in this case

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import json 
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

def load_data(data_path: str = "../data/processed_data.json", split_ratio: float = 0.8):
    """ 
    Load processed data
    """
    with open(data_path, 'r') as f:
        processed_data = json.load(f)
        
    # Train model and get predictions
    X = processed_data['prompt']
    y = np.array(processed_data['label'])
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_ratio, random_state=42)
        
    return X_train, X_test, y_train, y_test


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
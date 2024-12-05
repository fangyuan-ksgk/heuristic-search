# Trainable token on top of a LLM (Qwen-0.5B) in this case
# Temasek Foundation Dataset used here :: Binary + Comment type of data format

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, random
import numpy as np 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Union, List
from tqdm import tqdm as tqdm 
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "processed_data.json")
RUN_DIR = os.path.join(SCRIPT_DIR, "..", "runs")

def load_hf_model_precise(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """ 
    Load Huggingface model and tokenizer
    Load in model with same precision level (MPS and CUDA)
    """    
    if torch.backends.mps.is_available():
        torch_dtype = torch.float32
        device_map = "mps"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
    else:
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
        device_map = "cuda"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,  # works (I think there are other options?)
            use_cache=True # enable KV cache
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer



def load_tf_data(data_path: str = BASE_PATH, split_ratio: float = 0.8):
    """ 
    Load processed data -- let's add some phony data in-place
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    # train-test split with fixed seed
    train_indices, test_indices = train_test_split(range(len(data['prompt'])), test_size=0.2, random_state=42)
    train_data = {k: [v[i] for i in train_indices] for k, v in data.items()}
    test_data = {k: [v[i] for i in test_indices] for k, v in data.items()}
        
    return train_data, test_data 




def tf_label_metric(target_labels: Union[str, list], pred_labels: Union[str, list],
                    beta: float = 5):
    """ 
    "No" prediction > "Yes" prediction 
    Recall > Precision
    Weighted score (recall + precision) with beta : 1 as weights 
    """
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    if isinstance(pred_labels, str): 
        pred_labels = [pred_labels]
        
    # Convert labels to lowercase for case-insensitive comparison
    target_labels = [label.lower() for label in target_labels]
    pred_labels = [label.lower() for label in pred_labels]
    
    # For single label comparison, we can simplify:
    # true positive: target is "no" and prediction is "no"
    true_pos = sum(1 for t, p in zip(target_labels, pred_labels) if t == "no" and p == "no")
    
    # Calculate precision and recall
    # If target is "no", recall denominator is 1, else 0
    # If pred is "no", precision denominator is 1, else 0
    no_targets = sum(1 for t in target_labels if t == "no")
    no_preds = sum(1 for p in pred_labels if p == "no")
    
    no_precision = true_pos / no_preds if no_preds > 0 else 0
    no_recall = true_pos / no_targets if no_targets > 0 else 0
    
    # Simple weighted score using beta
    weighted_score = ((1 / (1 + beta)) * no_precision + (beta / (1 + beta)) * no_recall) if (no_precision + no_recall) > 0 else 0.0
    
    err_msg = []
    for t, p in zip(target_labels, pred_labels):
        if t != p: 
            err_msg.append(f"Target is '{t}', but prediction is '{p}'") 
        
    return weighted_score, "\n".join(err_msg) if err_msg else ""

tf_metric_map = {"label": tf_label_metric}


# Worth including diversity in response template
RESPONSE_TEMPLATES = [
    '```json\n{\n    "decision": "{label}",\n    "comment": "{comment}"\n}\n```',
    '```json\n{\n    "comment": "{comment}",\n    "decision": "{label}"\n}\n```'
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
            prefix_mask = torch.ones(batch_size, self.n_learnable_tokens, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
        # Adjust labels if provided
        if labels is not None: 
            prefix_labels = torch.full((batch_size, self.n_learnable_tokens), -100, dtype=labels.dtype, device=labels.device)
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
        else:
            return outputs
        
    def generate(self, prompts: Union[str, List[str]], max_new_tokens: int = 100):
        """
        Generate text using the model with soft prompts prepended.
        
        Args:
            prompts: Single prompt string or list of prompt strings
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            List[str]: List of generated texts
        """
        # Handle single prompt case
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Tokenize the input prompts
        tokenized = self.tokenizer(prompts, 
                                 return_tensors="pt", 
                                 padding=True,
                                 truncation=True).to(self.device)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        batch_size = input_ids.shape[0]
        
        # Get input embeddings for the prompts
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Expand soft prompts for batch size
        soft_prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate soft prompts with input embeddings
        inputs_embeds = torch.cat([soft_prompts, inputs_embeds], dim=1)
        
        # Adjust attention mask
        prefix_mask = torch.ones(batch_size, self.n_learnable_tokens, device=self.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Generate with the modified inputs
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode the generated tokens for each sequence in batch
        generated_texts = []
        for output in outputs:
            # Skip the soft prompt tokens for each sequence
            text = self.tokenizer.decode(
                output[self.n_learnable_tokens:],
                skip_special_tokens=True
            )
            generated_texts.append(text)
        
        # If input was single prompt, return single result
        if len(generated_texts) == 1:
            return generated_texts[0]
        
        return generated_texts

def check_and_handle_nans(loss, model, pred_logits, batch_idx, optimizer):
    """
    Check for NaN/Inf values and handle recovery
    Returns: bool indicating if training should continue
    """
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\nNaN/Inf loss detected in batch {batch_idx}!")
        print(f"Device: {model.device}, Dtype: {next(model.parameters()).dtype}")
        print(f"Soft prompts stats: min={model.soft_prompts.min():.4f}, max={model.soft_prompts.max():.4f}")
        print(f"Logits stats: min={pred_logits.min():.4f}, max={pred_logits.max():.4f}")
        print(f"Loss value: {loss.item()}")
        
        # Clear gradients
        optimizer.zero_grad()
        
        # If this happens in first batch, we should stop training
        if batch_idx == 0:
            return False
            
        # For later batches, skip this batch and continue training
        return True
        
    return True

import re 
import ast 

def extract_json_from_text(text):
    """
    Extracts a JSON object from a text containing either a JSON code block or a JSON-like structure.
    
    Parameters:
        text (str): The input text containing the JSON code block or JSON-like structure.
        
    Returns:
        dict: The parsed JSON object.
        
    Raises:
        ValueError: If no JSON structure is found or JSON is invalid.
    """
    # Available Patterns
    code_json_pattern = r'```json\s*(\{.*?\})\s*```'
    code_python_pattern = r'```python\s*(.*?)\s*```'
    json_list_pattern = r'```json\s*(.*?)\s*```'
    json_dict_pattern = r'\{[^}]+\}'
    
    code_json_match = re.search(code_json_pattern, text, re.DOTALL)
    code_python_match = re.search(code_python_pattern, text, re.DOTALL)
    list_match = re.search(json_list_pattern, text, re.DOTALL)
    dict_match = re.search(json_dict_pattern, text, re.DOTALL)
    
    if code_json_match:
        json_str = code_json_match.group(1)
    elif code_python_match:
        json_str = code_python_match.group(1)
    elif list_match:
        json_str = list_match.group(1)
    elif dict_match:
        json_str = dict_match.group(0)
    else:
        raise ValueError("No JSON structure found in the provided text.")
          
    # return json_str
    # json_str = json_str.replace("'", '"')
    error_msg = ""
    try:
        json_data = json.loads(json_str)
        return json_data 
    except json.JSONDecodeError as e:
        error_msg += f"JsonDecodeError : \n{e}"
    try:
        json_data = ast.literal_eval(json_str)
        return json_data
    except Exception as e:
        error_msg += f"AstLiteralError : \n{e}"
        
    raise ValueError(error_msg)


def map_pred_label(pred_label: str): 
    """ 
    Many funny label could be generated, we just treat them as rejection as long as no 'yes' is involved
    """
    if "yes" in pred_label: 
        return "yes"
    elif "accepted" in pred_label:
        return "yes"
    else:
        return "no"
    
def map_target_label(target_label: str):
    if "maybe" in target_label:
        return "yes"
    elif "yes" in target_label:
        return "yes"
    else:
        return "no"


def _label_metric(target_label: str, generated_response: str) -> float: 
    """ 
    Evaluate the generated response against the target label
    """
    try:
        parsed_response = extract_json_from_text(generated_response)
    except Exception as e:
        return 0.0, f"Error parsing generated response: {e}"
    
    if 'decision' not in parsed_response:
        return 0.0, "Generated response does not contain 'decision' key"
    else:
        generated_label = parsed_response["decision"]
        
        generated_label = map_pred_label(generated_label.lower())
        target_label = map_target_label(target_label.lower())
        
        if generated_label.lower() != target_label.lower():
            return 0.0, f"Target label is '{target_label}', but generated label is '{generated_label}'"
        else:
            return 1.0, "Perfect!"
        
def label_metric(target_labels: list, generated_responses: list) -> tuple[float, list]:
    """
    Evaluate the generated responses against the target labels
    """
    scores = []
    err_msgs = []
    for target_label, generated_response in zip(target_labels, generated_responses):
        score, err_msg = _label_metric(target_label, generated_response)
        scores.append(score)
        err_msgs.append(err_msg)
    avg_score = sum(scores) / len(scores)
    
    return avg_score, err_msgs
        

def test_model(model_with_soft_prompt, tokenizer, test_data, batch_size=24):
    data_len = len(test_data["prompt"])
    scores = []
    err_msgs = []  # Added to collect error messages

    for i in tqdm(range(0, data_len, batch_size), desc="Testing model"):
        prompts = test_data["prompt"][i:i+batch_size]
        labels = test_data["label"][i:i+batch_size]
        comments = test_data["comment"][i:i+batch_size]
        
        query_prompts = []
        for prompt, comment, label in zip(prompts, comments, labels):
            query_prompt, response_prompt = format_prompt_instruction_tuned(prompt, comment, label, tokenizer, previous_messages=[])
            query_prompts.append(query_prompt)
            
        generated_responses = model_with_soft_prompt.generate(query_prompts, max_new_tokens=600)
        
        for response, label in zip(generated_responses, labels):
            score, err_msg = label_metric(response, label)  # Now properly unpacking both return values
            scores.append(score)
            err_msgs.append(err_msg)
            
        break # one-batch as validation set

    avg_score = sum(scores) / len(scores)
    print(f"Average score: {avg_score}")
    
    # Print error summary if needed
    error_count = sum(1 for msg in err_msgs if msg)
    if error_count > 0:
        print(f"\nFound {error_count} errors out of {len(err_msgs)} predictions")
    
    return avg_score, err_msgs  # Return both score and error messages for analysis



def train_soft_prompt(model, train_dataloader, num_epochs=5, learning_rate=1e-4, accumulation_steps=4, print_info: bool = True):
    """ 
    Train soft prompt without mixed precision training
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Add running average window
    running_window = 50
    running_losses = []
    
    # Add early stopping parameters
    best_metric = 0.0
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_losses = []
        accumulated_loss = 0  # Track loss across accumulation steps
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Loss: N/A")
        optimizer.zero_grad()  # Zero gradients at start
        
        for batch_idx, batch in enumerate(progress_bar):

            # Move batch to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs, pred_logits, labels = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_pred_n_label=True
            )
            loss = outputs.loss / accumulation_steps  # Scale loss for accumulation
            
            # NaN checking
            should_continue = check_and_handle_nans(loss, model, pred_logits, batch_idx, optimizer)
            if not should_continue:
                raise ValueError("NaN/Inf loss detected in first batch - training stopped")
            elif should_continue is True and torch.isnan(loss):
                continue
            
            # Accumulate loss and gradients
            loss.backward()
            accumulated_loss += loss.item()
            
            # Only update weights after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()  # Reset gradients after update
                
                # Log the accumulated loss
                current_loss = accumulated_loss
                accumulated_loss = 0  # Reset accumulator

            # More conservative soft prompt value clipping
            with torch.no_grad():
                model.soft_prompts.data.clamp_(-0.5, 0.5)
            
            current_loss = loss.item() * accumulation_steps
            total_loss += current_loss
            epoch_losses.append(current_loss)
            running_losses.append(current_loss)
            
            # Calculate and display running average loss
            if len(running_losses) > running_window:
                running_losses.pop(0)
            running_avg = sum(running_losses) / len(running_losses)
            
            # Update progress bar description with current loss
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_avg:.4f}")
            
            # Update progress bar with current losses
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{running_avg:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / len(train_dataloader)
        
        if print_info:
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Min loss: {min(epoch_losses):.4f}")
            print(f"Max loss: {max(epoch_losses):.4f}")
        
        scheduler.step(avg_loss)
    
    return model

        
#########################################################################
# Prompt Tuning HF                                                      #
# Turns out to have same speed as token tuning, while being less stable #
#########################################################################

from peft import PromptEmbedding, PromptTuningConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup

def setup_prompt_tuning_config(config_dict: dict) -> PromptTuningConfig:
    config = PromptTuningConfig(
            peft_type="PROMPT_TUNING",
            task_type="CAUSAL_LM",
            num_virtual_tokens=config_dict["num_virtual_tokens"],
            token_dim=config_dict["token_dim"],
            num_transformer_submodules=1,
            num_attention_heads=config_dict["num_attention_heads"],
            num_layers=config_dict["num_layers"],
            prompt_tuning_init="TEXT",
        prompt_tuning_init_text=config_dict["prompt_tuning_init_text"],
        tokenizer_name_or_path=config_dict["model_name"],
    )
    return config

def setup_model_for_training(
    base_model,
    config: PromptTuningConfig,
    learning_rate: float,
    num_epochs: int,
    train_dataloader,
    device: str = "cuda",
) -> tuple:
    """Setup model with prompt tuning and optimization components"""
        
    model = get_peft_model(base_model, config).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    return model, optimizer, lr_scheduler


def train_epoch(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    lr_scheduler,
    tokenizer,
    device: str = "cuda",
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Run one training epoch and evaluation with simplified gradient handling"""
    
    model.train()
    total_loss = 0
    valid_batches = 0  # Track number of valid batches
    optimizer.zero_grad()
    
    def handle_gradients(batch_idx: int, loss: torch.Tensor) -> bool:
        """Handle gradient updates and clipping. Returns False if we should skip this batch."""
        try:
            # Check if loss is valid before proceeding
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss value: {loss.item()} in batch {batch_idx}")
                optimizer.zero_grad()
                return False
            
            # First clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            # Check if gradients are too large even after clipping
            if torch.isnan(grad_norm) or grad_norm > max_grad_norm * 10:
                print(f"Gradient norm too large: {grad_norm}, skipping batch {batch_idx}")
                optimizer.zero_grad()
                return False
                
            # If everything is fine, step optimizer and scheduler
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            return True
            
        except Exception as e:
            print(f"Error in gradient handling: {e}")
            optimizer.zero_grad()
            return False
    
    # Training loop
    progress_bar = tqdm(train_dataloader, desc="Training")
    accumulated_loss = 0.0  # Initialize as float
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            # Ensure loss is a tensor and divide by accumulation_steps
            if not isinstance(accumulation_steps, (int, float)):
                accumulation_steps = int(accumulation_steps)
            loss = outputs.loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()  # Convert to float for accumulation
            
            # Update weights after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if handle_gradients(batch_idx, loss * accumulation_steps):  # Pass tensor loss
                    total_loss += accumulated_loss
                    valid_batches += 1
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{accumulated_loss:.4f}',
                        'avg_loss': f'{total_loss/max(1, valid_batches):.4f}'
                    })
                accumulated_loss = 0.0  # Reset as float
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            print(f"accumulation_steps type: {type(accumulation_steps)}")
            print(f"loss type: {type(outputs.loss)}")
            continue
    
    # Handle any remaining gradients
    if accumulated_loss > 0:
        if handle_gradients(batch_idx, torch.tensor(accumulated_loss, device=device)):
            total_loss += accumulated_loss
            valid_batches += 1
    
    # Calculate average loss only over valid batches
    avg_train_loss = total_loss / max(1, valid_batches)
    
    # Evaluation loop
    model.eval()
    total_eval_loss = 0.0  # Initialize as float
    valid_eval_batches = 0
    eval_preds = []
    
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_eval_loss += loss.item()  # Convert to float
                    valid_eval_batches += 1
                eval_preds.extend(
                    tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                )
        except Exception as e:
            print(f"Error in evaluation: {e}")
            continue
    
    avg_eval_loss = total_eval_loss / max(1, valid_eval_batches)
    
    print(f"\nEpoch Summary:")
    print(f"Valid training batches: {valid_batches}/{len(train_dataloader)}")
    print(f"Average training loss: {avg_train_loss:.4f}")
    print(f"Valid evaluation batches: {valid_eval_batches}/{len(test_dataloader)}")
    print(f"Average evaluation loss: {avg_eval_loss:.4f}")
    
    return avg_train_loss, avg_eval_loss



# test generation function 
def generate_text(prompts: Union[str, List[str]], 
                 model, 
                 tokenizer,
                 max_new_tokens: int = 600,
                 temperature: float = 0.9,
                 num_return_sequences: int = 1) -> list[str]:
    """
    Generate text using a pre-trained language model with batch processing.
    
    Args:
        prompts (Union[str, List[str]]): Single prompt or list of input prompts
        model: The model to use for generation
        tokenizer: The tokenizer to use
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Controls randomness in generation
        num_return_sequences (int): Number of sequences to generate per prompt
        
    Returns:
        list[str]: List of generated text sequences
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Handle single prompt case
    # Handle single prompt case
    if isinstance(prompts, str):
        prompts = [prompts]

    # Encode all prompts in batch
    inputs = tokenizer(prompts, 
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=True,
                    padding_side="left",
                    truncation=True).to("cuda")

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

    full_texts = [
        tokenizer.decode(output, skip_special_tokens=False)
        for output in outputs
    ]

    generated_texts = []
    for full_text, prompt in zip(full_texts, prompts):
        generated_text = full_text.split(prompt)[-1].split(tokenizer.eos_token)[0]
        generated_texts.append(generated_text)
    
    # If num_return_sequences > 1, reshape the output list
    if num_return_sequences > 1:
        # Reshape into [batch_size][num_return_sequences]
        generated_texts = [
            generated_texts[i:i+num_return_sequences] 
            for i in range(0, len(generated_texts), num_return_sequences)
        ]
    
    return generated_texts


# Main execution pipeline
def run_prompt_tuning_pipeline(
    config_dict: dict,
    num_epochs: int = 50,
    learning_rate: float = 3e-2,
    device: str = "cuda"
) -> tuple[object, list]:
    """Main pipeline for prompt tuning and evaluation"""

    # Load model and data
    print("Loading Model...")
    model_name = config_dict.get("model_name", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
    model, tokenizer = load_hf_model_precise(model_name)

    print("Loading Dataset ...")
    tf_dataset = ClsCommentDataset(load_tf_data, tokenizer, train=True)
    testset = ClsCommentDataset(load_tf_data, tokenizer, train=False)
    
    train_dataloader = DataLoader(tf_dataset, batch_size=config_dict["batch_size"], shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=config_dict["batch_size"], shuffle=True)

    # Setup training    
    config = setup_prompt_tuning_config(config_dict)
    
    model, optimizer, lr_scheduler = setup_model_for_training(
        model, config, learning_rate, num_epochs, train_dataloader, device
    )

    print("Start training loop ...")
    # Training loop
    for epoch in range(num_epochs):
        train_loss, eval_loss = train_epoch(
            model, train_dataloader, test_dataloader,
            optimizer, lr_scheduler, tokenizer, device, accumulation_steps=config_dict["accumulation_steps"]
        )
    
    # Evaluate final model
    print("Evaluating model performance ...")
    _, test_data = load_tf_data()
    generated_responses = evaluate_model_outputs(test_data, model, tokenizer)
    
    # save generated response & config dictionary
    config_id = config_dict["config_id"]+"_prompt_tuning"
    with open(f"{RUN_DIR}/generated_responses_{config_id}.json", "w") as f:
        json.dump(generated_responses, f)
    with open(f"{RUN_DIR}/config_{config_id}.json", "w") as f:
        json.dump(config_dict, f)
    
    return model, generated_responses


def evaluate_model_outputs(trained_model, tokenizer, cap_num: int = 30, config_dict: dict = None) -> list:
    
    generated_responses = []

    _, test_data = load_tf_data()

    with torch.no_grad():
        query_prompts = []
        
        for data in test_data["prompt"][:cap_num]:
            query_prompt, _ = format_prompt_instruction_tuned(
                data, 
                test_data["comment"][len(generated_responses)],  # get corresponding comment
                test_data["label"][len(generated_responses)],    # get corresponding label
                tokenizer, 
                previous_messages=[]
            )
            query_prompts.append(query_prompt)
            
        # batch prompts into group each of size 20 and conduct batch generations
        for i in range(0, len(query_prompts), 20):
            batch_prompts = query_prompts[i:i+20]
            if isinstance(trained_model, SoftPromptLLM):
                # SoftPrompt LLM has generation method 
                generated_responses.extend(trained_model.generate(batch_prompts, max_new_tokens=config_dict.get("max_new_tokens", 2024)))
            else:
                responses = generate_text(batch_prompts, trained_model, tokenizer)
                generated_responses.extend(responses)
                
        # assinging fitness score 
        fitness, err_msgs = label_metric(test_data["label"][:cap_num], generated_responses)
            
    return fitness, generated_responses, err_msgs


def run_token_tuning_pipeline(
    config_dict: dict, 
    num_epochs: int = 50, 
    learning_rate: float = 3e-2, 
    device: str = "cuda",
    cap_num: int = 30
) -> tuple[object, list]:
    """Main pipeline for token tuning and evaluation"""
    
    def print_section(title: str, char: str = "="):
        width = 80
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}\n")
        
    def print_metric_comparison(title: str, metrics: dict):
        print(f"\n{title}")
        print("-" * 60)
        for name, value in metrics.items():
            print(f"│ {name:<25} │ {value:>25} │")
        print("-" * 60)
    
    print_section("TOKEN TUNING PIPELINE", "═")
    print("Configuration:")
    print("├── Model:", config_dict.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    print("├── Config ID:", config_dict["config_id"])
    print("├── Epochs:", num_epochs)
    print("├── Learning Rate:", learning_rate)
    print("├── Batch Size:", config_dict.get("batch_size", 24))
    print("├── Device:", device)
    print("└── Evaluation Cap:", cap_num)

    # Load model & tokenizer
    print_section("MODEL INITIALIZATION", "─")
    print("🔄 Loading model and tokenizer...")
    model, tokenizer = load_hf_model_precise(config_dict.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    model = model.to(device)
    print("✅ Model loaded successfully")
    
    # Original Model Evaluation
    print("\n📊 Evaluating original model baseline...")
    orig_fitness, orig_responses, orig_err_msgs = evaluate_model_outputs(model, tokenizer, cap_num, config_dict={})
    orig_error_rate = len([msg for msg in orig_err_msgs if msg])/len(orig_err_msgs)
    print("✅ Baseline evaluation complete")

    # Load datasets
    print_section("DATASET PREPARATION", "─")
    tf_dataset = ClsCommentDataset(load_tf_data, tokenizer, train=True)
    testset = ClsCommentDataset(load_tf_data, tokenizer, train=False)
    
    print_metric_comparison("Dataset Statistics", {
        "Training Samples": len(tf_dataset),
        "Test Samples": len(testset),
        "Batch Size": config_dict.get("batch_size", 24),
        "Total Batches": len(tf_dataset) // config_dict.get("batch_size", 24)
    })

    # Initialize dataloaders
    train_dataloader = DataLoader(
        tf_dataset, 
        batch_size=config_dict.get("batch_size", 24), 
        shuffle=True
    )

    # Initialize soft-prompt model
    print_section("SOFT PROMPT INITIALIZATION", "─")
    n_tokens = config_dict.get("n_learnable_tokens", 3)
    print(f"🔄 Initializing {n_tokens} learnable tokens...")
    model_with_soft_prompt = SoftPromptLLM(
        model, 
        tokenizer, 
        n_learnable_tokens=n_tokens,
        initialize_from_vocab=config_dict.get("initialize_from_vocab", True)
    ).to(device)
    print("✅ Soft prompts initialized successfully")
    
    # Initial Soft-prompt Evaluation
    print("\n📊 Evaluating initial soft-prompt model...")
    init_fitness, init_responses, init_err_msgs = evaluate_model_outputs(model_with_soft_prompt, tokenizer, cap_num, config_dict)
    init_error_rate = len([msg for msg in init_err_msgs if msg])/len(init_err_msgs)
    print("✅ Initial soft-prompt evaluation complete")

    # Train model using the existing train_soft_prompt function
    trained_model = train_soft_prompt(
        model_with_soft_prompt,
        train_dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        accumulation_steps=config_dict.get("accumulation_steps", 4)
    )

    # Generate responses on test set
    token_fitness, token_responses, token_err_msgs = evaluate_model_outputs(trained_model, tokenizer, cap_num, config_dict)

    training_info = {
        "config_dict": config_dict,
        "orig_fitness": orig_fitness,
        "orig_responses": orig_responses,
        "orig_err_msgs": orig_err_msgs,
        "init_fitness": init_fitness,
        "init_responses": init_responses,
        "init_err_msgs": init_err_msgs,
        "token_fitness": token_fitness,
        "token_responses": token_responses,
        "token_err_msgs": token_err_msgs
    }
    
    config_id = config_dict["config_id"]+"_token_tuning"
    with open(f"{RUN_DIR}/training_info_{config_id}.json", "w") as f:
        json.dump(training_info, f)

    return trained_model, token_responses
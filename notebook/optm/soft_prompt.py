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


def load_hf_model_precise(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """ 
    Load Huggingface model and tokenizer
    Load in model with same precision level (MPS and CUDA)
    """    
    if torch.backends.mps.is_available():
        torch_dtype = torch.float32
        device_map = "mps"
    else:
        # For CUDA, use float32 instead of float16
        torch_dtype = torch.float32  # Changed from float16
        device_map = "cuda"
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer


def load_tf_data(data_path: str = "../data/processed_data.json", split_ratio: float = 0.8):
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


from typing import Union, Callable


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

def label_metric(target_label: str, generated_response: str) -> float: 
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
        if generated_label.lower() != target_label.lower():
            return 0.0, f"Target label is '{target_label}', but generated label is '{generated_label}'"


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
            
            # Print detailed loss every 100 steps
            if batch_idx % running_window == 0 and print_info:
                
                print(f"\nStep {batch_idx}")
                print(f"Current loss: {current_loss:.4f}")
                print(f"Running average (last {running_window} steps): {running_avg:.4f}")
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
                print(f"Soft prompts range: [{model.soft_prompts.min():.4f}, {model.soft_prompts.max():.4f}]")
        
        avg_loss = total_loss / len(train_dataloader)
        
        if print_info:
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Min loss: {min(epoch_losses):.4f}")
            print(f"Max loss: {max(epoch_losses):.4f}")
        
        scheduler.step(avg_loss)
    
    return model
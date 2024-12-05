qwen_05_config_dict = {     
    "num_virtual_tokens": 4,
    "token_dim": 1024,
    "num_attention_heads": 16,
    "num_layers": 24,
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt_tuning_init_text": "ksgk",
    "config_id": "qwen0.5B",
    "batch_size": 36,
    "accumulation_steps": 4
}

qwen_7_config_dict = {
    "num_virtual_tokens": 4,
    "token_dim": 4096,
    "num_attention_heads": 32,
    "num_layers": 32,
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "prompt_tuning_init_text": "ksgk",
    "config_id": "qwen7B",
    "batch_size": 9,
    "accumulation_steps": 24
}

llama_3_8b_config_dict = {
    "num_virtual_tokens": 4,
    "token_dim": 4096,
    "num_attention_heads": 32,
    "num_layers": 32,
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt_tuning_init_text": "ksgk",
    "config_id": "llama3.1_8B",
    "batch_size": 9,
    "accumulation_steps": 24
}
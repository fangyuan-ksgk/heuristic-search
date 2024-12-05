from optm.soft_prompt import load_tf_data, _form_query, _form_response, load_hf_model_precise, RUN_DIR
import pyreft, transformers
from pyreft import ReftTrainerForCausalLM
import torch, json
from .soft_prompt import RUN_DIR, load_hf_model_precise, evaluate_model_outputs, label_metric


# Loading moded
def load_reft_model(model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"): 
    
    model, tokenizer = load_hf_model_precise(model_name) # load model & tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, model_max_length=4096, 
                padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token


    reft_config = pyreft.ReftConfig(representations={
        "layer": 8, "component": "block_output",
        "low_rank_dimension": 4,
        "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
        low_rank_dimension=4)})

    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device("cuda")
    reft_model.print_trainable_parameters()
    
    return reft_model, tokenizer, model


def prepare_reft_training_examples(tokenizer, use_train: bool = True):
    msg = [{"role": "system", "content": "SYSTEM_CONTENT"}, {"role": "user", "content": "%s"}, {"role": "assistant", "content": "ASSISTANT_CONTENT"}]
    default_template = tokenizer.apply_chat_template(msg, tokenize=False)
    prompt_no_input_template = ("").join(default_template.split(tokenizer.eos_token)[1:]).split("ASSISTANT_CONTENT")[0]
    response_no_input_template = "%s" + default_template.split("ASSISTANT_CONTENT")[1]
    
    prompt_no_input_template, response_no_input_template
    
    train_set, test_set = load_tf_data()
    
    from optm.soft_prompt import _form_query, _form_response
    
    training_examples = []
    for prompt, label, comment in zip(train_set["prompt"], train_set["label"], train_set["comment"]):
        query_str = _form_query(prompt)
        response_str = _form_response(label, comment) 
        query = prompt_no_input_template % query_str
        response = response_no_input_template % response_str
        training_examples.append(tuple([query, response]))

    testing_examples = []
    for prompt, label, comment in zip(test_set["prompt"], test_set["label"], test_set["comment"]):
        query_str = _form_query(prompt)
        response_str = _form_response(label, comment) 
        query = prompt_no_input_template % query_str
        response = response_no_input_template % response_str
        testing_examples.append(tuple([query, response]))
    
    return training_examples, testing_examples


def train_reft_model(reft_model, training_examples, tokenizer, config_dict: dict): 
    
    data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer, reft_model, [e[0] for e in training_examples], # Used to be 'model' but replace with 'reft_model' 
        [e[1] for e in training_examples]
    )
    
    # train
    training_args = transformers.TrainingArguments(
        num_train_epochs=config_dict.get("num_epochs", 50),
        output_dir=f"{RUN_DIR}/reft",
        per_device_train_batch_size=config_dict.get("batch_size", 4),
        gradient_accumulation_steps=config_dict.get("accumulation_steps", 8),  # Added gradient accumulation
        learning_rate=config_dict.get("lr", 4e-3),
        logging_steps=config_dict.get("logging_steps", 2),
        report_to=[]
    )
    
    trainer = ReftTrainerForCausalLM(
        model=reft_model, tokenizer=tokenizer, args=training_args, **data_module)
    
    _ = trainer.train()
    

def train_reft_model_old(reft_model, training_examples, tokenizer, config_dict: dict):
    
    data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer, reft_model, [e[0] for e in training_examples], # Used to be 'model' but replace with 'reft_model' 
        [e[1] for e in training_examples])
    
    # train
    training_args = transformers.TrainingArguments(
        num_train_epochs=config_dict.get("num_epochs", 50),
        output_dir=f"{RUN_DIR}/reft",
        per_device_train_batch_size=config_dict.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config_dict.get("accumulation_steps", 8),  # Added gradient accumulation
        learning_rate=config_dict.get("lr", 4e-3),
        logging_steps=config_dict.get("logging_steps", 2),
        report_to=[]
    )

    # This CustomReftTrainer is causing memory overloading issue ? (Hard to believe ...)
    class CustomReftTrainer(ReftTrainerForCausalLM):
        def compute_loss(self, model, inputs, *args, **kwargs):
            kwargs.pop('num_items_in_batch', None)
            loss = super().compute_loss(model, inputs, *args, **kwargs)
            torch.cuda.empty_cache()  # Clear CUDA cache after loss computation
            return loss


    trainer = CustomReftTrainer(
        model=reft_model, tokenizer=tokenizer, args=training_args, **data_module)

    _ = trainer.train()
    
    
def get_reft_model_response(reft_model, tokenizer, prompt):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dict = tokenizer(prompt, return_tensors="pt").to(device)
    
    base_unit_location = input_dict["input_ids"].shape[-1] - 1  # last position
    _, reft_response = reft_model.generate(
        input_dict, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
        intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
        eos_token_id=tokenizer.eos_token_id, early_stopping=True
    )
    
    full_text = tokenizer.decode(reft_response[0], skip_special_tokens=False)
    return full_text.split(prompt)[-1]


def evaluate_reft_model_outputs(reft_model, tokenizer, cap_num: int = 30) -> tuple:
    _, test_set = load_tf_data()
    
    generated_responses = []
    true_responses = []
    
    with torch.no_grad():
        for prompt, label, comment in list(zip(test_set["prompt"], test_set["label"], test_set["comment"]))[:cap_num]:
            # Format prompt using existing helper functions
            query_str = _form_query(prompt)
            response_str = _form_response(label, comment)
            
            # Use the same template formatting as in prepare_reft_training_examples
            msg = [{"role": "system", "content": "SYSTEM_CONTENT"}, {"role": "user", "content": "%s"}, {"role": "assistant", "content": "ASSISTANT_CONTENT"}]
            default_template = tokenizer.apply_chat_template(msg, tokenize=False)
            prompt_template = ("").join(default_template.split(tokenizer.eos_token)[1:]).split("ASSISTANT_CONTENT")[0]
            
            # Format the final prompt
            query_prompt = prompt_template % query_str
            
            # Generate response using REFT model
            response = get_reft_model_response(reft_model, tokenizer, query_prompt)
            generated_responses.append(response)
            true_responses.append(response_str)
    
    # Calculate fitness scores
    from .soft_prompt import label_metric
    fitness, err_msgs = label_metric(test_set["label"][:cap_num], generated_responses)
            
    return fitness, generated_responses, err_msgs



def run_reft_pipeline(
    config_dict: dict, 
    num_epochs: int = 50, 
    learning_rate: float = 3e-2, 
    device: str = "cuda",
    cap_num: int = 30
) -> tuple[object, list]:
    """Main pipeline for REFT tuning and evaluation"""
    
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
    
    # Get consistent batch size with a default of 5 for REFT
    batch_size = config_dict.get("batch_size", 5)  # Changed default from 8 to 5
    
    print_section("REFT TUNING PIPELINE", "═")
    print("Configuration:")
    print("├── Model:", config_dict.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    print("├── Config ID:", config_dict["config_id"])
    print("├── Epochs:", num_epochs)
    print("├── Learning Rate:", learning_rate)
    print(f"├── Batch Size: {batch_size}")  # Using the consistent batch_size
    print("├── Device:", device)
    print("└── Evaluation Cap:", cap_num)    

    # Load REFT model & tokenizer
    print_section("MODEL INITIALIZATION", "─")
    print("🔄 Loading REFT model and tokenizer...")
    reft_model, tokenizer, model = load_reft_model(config_dict.get("model_name"))
    print("✅ REFT model loaded successfully")
    
    # Prepare training examples
    print_section("DATASET PREPARATION", "─")
    print("🔄 Preparing training and testing examples...")
    training_examples, testing_examples = prepare_reft_training_examples(tokenizer)
    
    # # Evaluate Base model
    # print_section("BASE MODEL EVALUATION", "─")
    # print("🔄 Evaluating base model...")
    # from .soft_prompt import evaluate_model_outputs
    # orig_fitness, orig_responses, orig_err_msgs = evaluate_model_outputs(model, tokenizer, cap_num, config_dict={})
    # print("✅ Base model evaluation complete")
    
    print_metric_comparison("Dataset Statistics", {
        "Training Samples": len(training_examples),
        "Test Samples": len(testing_examples),
        "Batch Size": batch_size,  # Using the consistent batch_size
    })

    # Train the REFT model
    print_section("REFT MODEL TRAINING", "─")
    print("🔄 Training REFT model...")
    train_reft_model(reft_model, training_examples, tokenizer, config_dict)
    print("✅ REFT training complete")
    
    # Evaluate the REFT model
    print_section("REFT MODEL EVALUATION", "─")
    print("🔄 Evaluating REFT model...")
    fitness, generated_responses, err_msgs = evaluate_reft_model_outputs(reft_model, tokenizer, cap_num)
    print("✅ REFT evaluation complete")
    
    training_info = {
        "config_dict": config_dict,
        # "orig_fitness": orig_fitness,
        # "orig_responses": orig_responses,
        # "orig_err_msgs": orig_err_msgs,
        "reft_fitness": fitness,
        "reft_responses": generated_responses,
        "reft_err_msgs": err_msgs
    }
    
    config_id = config_dict["config_id"]+"_reft"
    with open(f"{RUN_DIR}/training_info_{config_id}.json", "w") as f:
        json.dump(training_info, f)

    return reft_model, testing_examples
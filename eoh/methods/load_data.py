from .meta_prompt import MetaPrompt, PromptMode

def prep_tf_node(prompt_mode: bool = True):
    """
    2nd ver. bool object for label (care about No and Non-No -- yes & maybe)
    """
    import sys 
    sys.path.append("../tuning/") 
    from optm.soft_prompt import load_tf_data, BASE_PATH
    train_data, test_data = load_tf_data(BASE_PATH)
    
    mode = PromptMode.PROMPT if prompt_mode else PromptMode.CODE
    
    tf_meta_prompt = MetaPrompt(
        task = "Evaluate grant application, make a decision (Yes, No, Maybe) and a brief comment explanating your decision on why this project is likely to be accepted or rejected.",
        func_name = "evaluate_grant",
        inputs = ["project_description"],
        outputs = ["decision", "comment"],
        input_types = ["str"],
        output_types = ["bool", "str"],
        mode = mode
    )

    # Prepare test cases :: input dict & output dict
    def map_label(label: str) -> bool: 
        if label.lower() in ["yes", "maybe"]: 
            return True 
        return False 
    
    test_cases = []
    for prompt, label, comment in zip(train_data["prompt"], train_data["label"], train_data["comment"]):
        test_cases.append(({"project_description": prompt}, {"decision": map_label(label), "comment": comment}))
        
    return tf_meta_prompt, test_cases


# Custom Metric for Temasek Foundation dataset

def map_decision_to_bool(decision: str) -> bool:
    yes_list = ["yes", "approve", "accept", "approved", "true"]
    no_list = ["no", "reject", "decline", "rejected", "false"]
    if decision in yes_list:
        return True
    elif decision in no_list:
        return False
    else:
        return True
    
def compare_decision(pred_decision: str, target_decision: bool) -> bool:
    
    if isinstance(pred_decision, str):
        pred_decision = map_decision_to_bool(pred_decision.lower())
    elif isinstance(pred_decision, bool):
        pred_decision = pred_decision
    else:
        pred_decision = False
            
    is_aligned = pred_decision == target_decision
    if not is_aligned:
        err_msg = f"Decision mismatch: {pred_decision} (pred) != {target_decision} (target)"
    else:
        err_msg = ""
    return is_aligned, err_msg
    
custom_metric_map = {"decision": compare_decision}
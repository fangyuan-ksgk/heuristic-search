from .meta_prompt import MetaPrompt, PromptMode

def prep_tf_node(prompt_mode: bool = True):
    """ 
    Prepare test cases, meta prompt, and metric map for Temasek Foundation dataset
    """
    
    import sys 
    sys.path.append("../notebook/") 
    from optm.soft_prompt import load_tf_data, tf_metric_map
    train_data, test_data = load_tf_data("../data/processed_data_clean.json")
    
    mode = PromptMode.PROMPT if prompt_mode else PromptMode.CODE
    
    tf_meta_prompt = MetaPrompt(
        task = "Evaluate grant application, make a decision (Yes, No, Maybe) and a brief comment explanating your decision on why this project is likely to be accepted or rejected.",
        func_name = "evaluate_grant",
        inputs = ["project_description"],
        outputs = ["label", "comment"],
        input_types = ["str"],
        output_types = ["str", "str"],
        mode = mode
    )

    # Prepare test cases :: input dict & output dict
    test_cases = []
    for prompt, label, comment in zip(train_data["prompt"], train_data["label"], train_data["comment"]):
        test_cases.append(({"project_description": prompt}, {"label": label, "comment": comment}))
    
    return tf_meta_prompt, test_cases, tf_metric_map
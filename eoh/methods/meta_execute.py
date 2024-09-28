import types
import importlib.util
import sys
from .meta_prompt import extract_json_from_text

def call_func_code(input_data, code: str, func_name: str, file_path=None):
    """ 
    Dynamic calling function defined in 'code' snippet
    - with support of external python file from 'file_name'
    - dynamic module used to cache the code-snippet
    """
    if file_path:
        # Load code from external file
        module_name = f"dynamic_module_{hash(file_path)}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    else:
        # Use the existing code string approach
        mod = types.ModuleType('dynamic_module')
        exec(code, mod.__dict__)
    
    # Get the function from the module
    if func_name not in mod.__dict__:
        raise ValueError(f"Function '{func_name}' not found in generated code")
    
    func = mod.__dict__[func_name]
    
    # Call the function with the input data
    return func(input_data)

def call_func_prompt(input_data, code: str, get_response: callable):
    """ 
    Prompt EvolNode forward propagation
    - Compile prompt with LLM and return the response
    """
    mod = types.ModuleType('dynamic_module')
    exec(code, mod.__dict__)
    func_name = "generate_prompt"
    prompt_func = mod.__dict__[func_name]
    prompt = prompt_func(input_data)
    response = get_response(prompt)
    try:
        output_dict = extract_json_from_text(response)
    except:
        output_dict = {}
    return output_dict
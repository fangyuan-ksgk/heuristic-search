import types
import importlib.util
import sys
from .meta_prompt import extract_json_from_text
from typing import Any, get_type_hints, Dict
import inspect


def call_func_code(input_data: Dict[str, Any], code: str, func_name: str, file_path: str = None) -> Any:
    """ 
    Dynamic calling function defined in 'code' snippet
    - with support of external python file from 'file_name'
    - dynamic module used to cache the code-snippet
    - supports multiple inputs as keyword arguments
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
    
    # Check input types
    type_hints = get_type_hints(func)
    if 'return' in type_hints:
        expected_return_type = type_hints['return']
        del type_hints['return']
    else:
        expected_return_type = Any

    # Check if all required parameters are provided
    sig = inspect.signature(func)
    required_params = {
        name: param for name, param in sig.parameters.items()
        if param.default == inspect.Parameter.empty and param.kind != inspect.Parameter.VAR_KEYWORD
    }
    if not all(name in input_data for name in required_params):
        missing = set(required_params) - set(input_data)
        raise ValueError(f"Missing required input parameters: {', '.join(missing)}")

    # Check input types
    for param_name, expected_type in type_hints.items():
        if param_name in input_data:
            actual_value = input_data[param_name]
            if not isinstance(actual_value, expected_type) and expected_type != Any:
                raise TypeError(f"Input data type mismatch for parameter '{param_name}'. Expected {expected_type}, got {type(actual_value)}")

    # Call the function with the input data
    result = func(**input_data)

    # Check output type
    if not isinstance(result, expected_return_type) and expected_return_type != Any:
        raise TypeError(f"Output data type mismatch. Expected {expected_return_type}, got {type(result)}")

    return result



# def call_func_code(input_data, code: str, func_name: str, file_path=None):
#     """ 
#     Dynamic calling function defined in 'code' snippet
#     - with support of external python file from 'file_name'
#     - dynamic module used to cache the code-snippet
#     """
#     if file_path:
#         # Load code from external file
#         module_name = f"dynamic_module_{hash(file_path)}"
#         spec = importlib.util.spec_from_file_location(module_name, file_path)
#         mod = importlib.util.module_from_spec(spec)
#         sys.modules[module_name] = mod
#         spec.loader.exec_module(mod)
#     else:
#         # Use the existing code string approach
#         mod = types.ModuleType('dynamic_module')
#         exec(code, mod.__dict__)
    
#     # Get the function from the module
#     if func_name not in mod.__dict__:
#         raise ValueError(f"Function '{func_name}' not found in generated code")
    
#     func = mod.__dict__[func_name]
    
#     # Call the function with the input data
#     return func(input_data)


def call_func_prompt(input_data: Dict[str, Any], code: str, get_response: callable):
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


def compile_check(code: str, input_data: Dict[str, Any], expected_output_type: Any) -> Tuple[bool, str]:
    """
    Check the fitness of the node by verifying compilation success and input/output correctness.
    Returns a tuple of (is_fit: bool, error_message: str)
    """
    # Check compilation success
    try:
        compile(code, '<string>', 'exec')
    except Exception as e:
        return False, f"Compilation Error: {str(e)}"

    # Check input/output type consistency
    try:
        result = call_func_code(input_data, code, 'main')
        if not isinstance(result, expected_output_type) and expected_output_type != Any:
            return False, f"Output type mismatch. Expected {expected_output_type}, got {type(result)}"
    except TypeError as e:
        return False, f"Type Error: {str(e)}"
    except Exception as e:
        return False, f"Runtime Error: {str(e)}"

    return True, "Compilation and type check successful"
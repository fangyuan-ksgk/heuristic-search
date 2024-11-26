import types
import importlib.util
import sys
from .meta_prompt import extract_json_from_text
from typing import Any, get_type_hints, Dict, List
import inspect
from typing import get_origin, get_args
import ast 
import astor 
import signal
import re 
from collections import defaultdict


def check_type(value, expected_type):
    if expected_type is Any:
        return True
    
    origin = get_origin(expected_type)
    if origin is None:
        # For non-generic types
        return isinstance(value, expected_type)
    else:
        # For generic types
        if not isinstance(value, origin):
            return False
        
        args = get_args(expected_type)
        if not args:
            return True  # No arguments to check
        
        if isinstance(value, (list, tuple, set)):
            return all(check_type(item, args[0]) for item in value)
        elif isinstance(value, dict):
            return (all(check_type(k, args[0]) for k in value.keys()) and
                    all(check_type(v, args[1]) for v in value.values()))
        else:
            # For other types of generics, you might need to add more specific checks
            return True
    

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out (> 3 seconds)")

def _call_func_code(input_data: Dict[str, Any], code: str, func_name: str, file_path: str = None) -> Any:
    """ 
    Dynamic calling function defined in 'code' snippet with 3-second timeout
    """
    # Set up the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3)  # Set timeout to 3 seconds
    
    try:
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
                if not check_type(actual_value, expected_type) and expected_type != Any:
                    raise TypeError(f"Input data type mismatch for parameter '{param_name}'. Expected {expected_type}, got {type(actual_value)}")

        # Call the function with the input data
        result = func(**input_data)

        # Check output type
        if expected_return_type != Any and not check_type(result, expected_return_type):
            raise TypeError(f"Output data type mismatch. Expected {expected_return_type}, got {type(result)}")

        # Disable the alarm
        signal.alarm(0)
        return result
        
    finally:
        # Ensure the alarm is disabled even if an error occurs
        signal.alarm(0)

def call_func_code(input_data: Dict[str, Any], code: str, func_name: str, file_path: str = None) -> Any:
    """ 
    With Error Message Output
    """
    try:
        return _call_func_code(input_data, code, func_name, file_path), ""
    except Exception as e:
        return None, str(e)


def _call_func_prompt(input_data: Dict[str, Any], code: str, get_response: callable):
    """ 
    Prompt EvolNode forward propagation
    - Compile prompt with LLM and return the response
    """
    # Set up the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3)  # Set timeout to 3 seconds
    
    try:
        mod = types.ModuleType('dynamic_module')
        exec(code, mod.__dict__)
        func_name = "generate_prompt"
        prompt_func = mod.__dict__[func_name]
        prompt = prompt_func(**input_data)
        print(prompt)
        response = get_response(prompt)
        print(response)
        
        try:
            output_dict = extract_json_from_text(response)
            # Disable the alarm
            signal.alarm(0)
            return output_dict
        
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
            
    finally:
        # Ensure the alarm is disabled even if an error occurs
        signal.alarm(0)

def call_func_prompt(input_data: Dict[str, Any], code: str, get_response: callable):
    """ 
    With Error Message Output
    """
    try:
        return _call_func_prompt(input_data, code, get_response), ""
    except Exception as e:
        return None, str(e)


class Timeout_:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, *args):
        signal.alarm(0)
        
        
def call_func_prompt_parallel(input_dicts: List[Dict[str, Any]], codes: List[str], max_tries: int, get_response: callable):
    """ 
    Parallel calling of prompt function
    """
    outputs_per_code_per_test = defaultdict(lambda: defaultdict(list))
    errors_per_code_per_test = defaultdict(lambda: defaultdict(list))
    
    prompts = []
    input_indices = []
    for (input_index, input_dict) in enumerate(input_dicts):
        for (code_index, code) in enumerate(codes):
            mod = types.ModuleType('dynamic_module')
            try:
                with Timeout_(3):  # 3-second timeout
                    exec(code, mod.__dict__)
                    func_name = "generate_prompt"
                    assert func_name in mod.__dict__, f"Function {func_name} not found in code #{code_index}"
                    prompt_func = mod.__dict__[func_name]
                    prompt = prompt_func(**input_dict)
                    prompts.append(prompt)
                    input_indices.append((input_index, code_index))
            except Exception as e:
                errors_per_code_per_test[code_index][input_index].append(str(e))
            
    responses = get_response(prompts)
    
    for (response, (input_index, code_index)) in zip(responses, input_indices):
        try:
            output_dict = extract_json_from_text(response)
            outputs_per_code_per_test[code_index][input_index].append(output_dict)
        
        except Exception as e:
            errors_per_code_per_test[code_index][input_index].append(f"Failed to parse LLM response -- " + str(e))
    
    output_per_code_per_test = defaultdict(lambda: defaultdict(dict))
    for code_index in outputs_per_code_per_test:
        for input_index in outputs_per_code_per_test[code_index]:
            for output_dict in outputs_per_code_per_test[code_index][input_index]:
                if output_dict != {}:   # Keep non-empty output dict
                    output_per_code_per_test[code_index][input_index] = output_dict
                
    return output_per_code_per_test, errors_per_code_per_test



def clean_up_ast_tree(new_tree):
    # Sort imports and remove duplicates
    import_nodes = []
    other_nodes = []
    
    for node in new_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not any(astor.to_source(node) == astor.to_source(existing) for existing in import_nodes):
                import_nodes.append(node)
        else:
            other_nodes.append(node)
    
    # Sort import nodes
    import_nodes.sort(key=lambda x: x.names[0].name if isinstance(x, ast.Import) else x.module)
    
    # Reconstruct the AST with sorted imports at the top
    new_tree.body = import_nodes + other_nodes
    
    return new_tree


def include_func_check(original_code, referrable_function_dict):
    """ 
    The AST parsing somehow misses out function call, this is to patch that issue with regex
    """
    need_include_func = []
    for (func_name, func_code) in referrable_function_dict.items():
        func_call_pattern = rf"(?<!def\s){re.escape(func_name)}\s*\("
        if re.search(func_call_pattern, original_code):
            # Function is called in the code, so we need to include its definition
            need_include_func.append(func_name)
    return need_include_func


def compile_code_with_references(node_code, referrable_function_dict):
    """ 
    Compile code with references to other functions
    """
    tree = ast.parse(node_code)
    
    added_functions = set()
    
    class ReferenceReplacer(ast.NodeTransformer):
        def visit_Import(self, node):
            return None if any(alias.name in referrable_function_dict for alias in node.names) else node
        
        def visit_ImportFrom(self, node):
            return None if node.module in referrable_function_dict else node
        
        def visit_FunctionDef(self, node):
            if node.name in referrable_function_dict and node.name not in added_functions:
                new_func = ast.parse(referrable_function_dict[node.name]).body[0]
                added_functions.add(node.name)
                return new_func
            return node            
        
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in referrable_function_dict and func_name not in added_functions:
                    added_functions.add(func_name)
            return node

    new_tree = ReferenceReplacer().visit(tree)
    
    # Add all referenced functions to the beginning of the tree
    for func_name in added_functions:
        new_func_tree = ast.parse(referrable_function_dict[func_name])
        new_tree.body = new_func_tree.body + new_tree.body
        
    # Patch
    patch_functions = include_func_check(node_code, referrable_function_dict)
    for func_name in patch_functions:
        if func_name not in added_functions:
            added_functions.add(func_name)
            new_func_tree = ast.parse(referrable_function_dict[func_name])
            new_tree.body = new_func_tree.body + new_tree.body

    new_tree = clean_up_ast_tree(new_tree)

    new_code = astor.to_source(new_tree)
    return new_code

def combine_scores(llm_scores, metric_scores):
    # Initialize combined scores dictionary
    combined_scores = defaultdict(lambda: defaultdict(list))
    combined_score = defaultdict(lambda: defaultdict(float))
    # For each code index in metric scores
    for k in metric_scores: 
        for i in metric_scores[k]: 
            combined_scores[k][i] += metric_scores[k][i]
    for k in llm_scores: 
        for i in llm_scores[k]: 
            combined_scores[k][i] += llm_scores[k][i] if type(llm_scores[k][i]) == list else [llm_scores[k][i]]
    
    for k in combined_scores: 
        for i in combined_scores[k]: 
            combined_score[k][i] = sum(combined_scores[k][i]) / len(combined_scores[k][i])
    
    return combined_score

def combine_errors(dict1, dict2):
    # Create a new defaultdict with the same nested structure
    combined = defaultdict(list)
    
    # Combine all errors from both dictionaries
    for dict_to_process in [dict1, dict2]:
        for code_index in dict_to_process:
            for test_index in dict_to_process[code_index]:
                combined[code_index].extend(dict_to_process[code_index][test_index])
    
    return combined
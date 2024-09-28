import types
import importlib.util
import sys

def call_func(input_data, code: str, func_name: str, file_path=None):
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
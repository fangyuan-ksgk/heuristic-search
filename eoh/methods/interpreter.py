import matplotlib.pyplot as plt
import subprocess
import re

def parse_python_code(response: str) -> str:
    code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    code_str = code_match.group(1).strip() if code_match else ""
    return code_str
    
def parse_response(response: str) -> str:
    return response.split("RESPONSE##")[-1]
    

class CodeInterpreter:
    def __init__(self):
        self.locals = {}
        self.globals = globals().copy()
        self.output = []
        self.figures = []
        
    def __call__(self, response: str):
        """ 
        Two situation: 
        - We have code-snippet, compile it, flag (True)
        - Otherwise, this is a direct response, return it with a flag (False)
        """
        code, is_code = self.parse_code(response)
        if is_code:
            self.execute(code)
            return self.return_results(), self.figure, True

        return parse_response(response), self.figure, False
        
        
    def parse_code(self, response: str):
        code = parse_python_code(response)
        if code:
            return code, True
        return response, False

    def execute(self, code):        
        # Reset output and figures
        self.output = []
        self.figures = []

        try:
            # Use subprocess to run the code
            result = subprocess.run(['python', '-c', code], capture_output=True, text=True, check=True)
            self.output.append(result.stdout)

            # TODO: Handle figure generation
            # This part needs to be adjusted as subprocess doesn't directly capture matplotlib figures

        except subprocess.CalledProcessError as e:
            pass
        except Exception as e:
            pass

    def get_output(self):
        return "\n".join(self.output)
            
    def return_results(self):
        compiled_result = self.get_output()
        return compiled_result
from .interpreter import CodeInterpreter
from typing import Callable
from tqdm import tqdm 
from .repo import *

def strategic_respond(json_file_path: str, 
                      img_base64: str,
                      user_question: str,
                      file_dag: dict,
                      interpreter: CodeInterpreter, 
                      get_claude_response: Callable):
    
    retrieved_info = ""
    progress_bar = tqdm(total=2, desc="Strategic response progress", unit="%")
    
    print("Coding to traverse file dependency graph...")
    prompt = get_navigate_prompt(json_file_path, user_question, retrieved_info, file_dag)
    response = get_claude_response(prompt, img=img_base64, img_type="image/png")
    code_result, _, _ = interpreter(response)
    print(f"Compiled code result: {code_result}")
    
    progress_bar.update(1)
    
    print("Forming final answer...")
    prompt = get_navigate_prompt(json_file_path, user_question, code_result, file_dag, use_code=False)
    response = get_claude_response(prompt, img=img_base64, img_type="image/png")
    final_response, figure, _ = interpreter(response)
    
    progress_bar.update(1)
    
    progress_bar.close()
    
    return final_response


class RepoAgent: 
    
    def __init__(self, temp_repo: str, get_vlm_response: Callable):
        self.temp_repo = temp_repo
        self.get_vlm_response = get_vlm_response
        self.file_dag = directory_to_file_dag(temp_repo)
        self.file_names = [self.file_dag[k]['name'] for k in self.file_dag]
        module_dag = parse_directory_dag(temp_repo)
        module_dag = bottom_up_summarization(module_dag, get_vlm_response) # Get a nice VLM to speed-up the inference, in case claude dies ...
        self.module_dag = module_dag
        self.module_names = [self.module_dag[k]['name'] for k in self.module_dag]
        self.file_dag_path = save_dag_as_json(self.module_dag, temp_repo)
        self.module_dag_path = save_dag_as_json(self.file_dag, temp_repo)
        
    def visualize_file(self):
        return visualize_dag(self.file_dag)
    
    def visualize_module(self):
        return visualize_dag(self.module_dag)
        
    @property 
    def module_graph(self):
        png_module_graph_path = self.visualize_module()
        return file_to_preprocessed_img(png_module_graph_path)
    
    @property 
    def file_graph(self):
        png_file_graph_path = self.visualize_file()
        return file_to_preprocessed_img(png_file_graph_path)
    
    def _respond(self, question: str):
        interpreter = CodeInterpreter()
        return strategic_respond(self.module_dag_path, self.module_graph, question, self.module_dag, interpreter, self.get_vlm_response) # Module DAG contains File DAG
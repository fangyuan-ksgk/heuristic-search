from dataclasses import dataclass
from enum import Enum

class PromptMode(Enum):
    CODE = "code"
    TOOL = "tool"
    PROMPT = "prompt"
    
# MetaPrompt describe meta-heuristic for each node's generation (i1) evolution (e1, e2) and mutation (m1, m2)
# - Code
# - Tool 
# - Prompt
@dataclass
class MetaPrompt:
    task: str
    func_name: str
    input: str
    output: str
    
    @property
    def joined_inputs(self):
        if len(self.input) > 1:
            return ", ".join("'" + s + "'" for s in self.input)
        else:
            return "'" + self.input[0] + "'"
        
    @property
    def joined_outputs(self):
        if len(self.output) > 1:
            return ", ".join("'" + s + "'" for s in self.output)
        else:
            return "'" + self.output[0] + "'"
        
    def _base_prompt(self, mode: PromptMode):
        if mode == PromptMode.CODE:
            prompt_content = f"First, describe your new algorithm and main steps in one sentence. "\
                "The description must be inside a brace. Next, implement it in Python as a function named "\
                f"{self.func_name}. This function should accept {len(self.input)} input(s): "\
                f"{self.joined_inputs}. The function should return {len(self.output)} output(s): "\
                f"{self.joined_outputs}. {self.inout_inf} "\
                f"{self.other_inf}\n"\
                "Do not give additional explanations."
            return prompt_content
        elif mode == PromptMode.PROMPT:
            prompt_content = f"First, describe your new reasoning and main thoughts in one sentence."\
                "The description must be inside a brace. Implement a Python function that generates a prompt to guide an AI in completing the task. "\
                f"Follow these specifications: - Function name: generate_prompt - Input parameters: {self.joined_inputs} - Return value: A string containing the final prompt for the AI.\n" \
                "Your function should incorporate the reasoning from step 1 and use the input parameters to create a tailored prompt for the task."
            return prompt_content
        elif mode == PromptMode.TOOL:
            raise NotImplementedError
        
    def _get_prompt_i1(self, mode: PromptMode):
        prompt_content = f"{self.task}\n{self._base_prompt(mode)}"
        return prompt_content
        
    def _get_prompt_e1(self, indivs: list, mode: PromptMode):
        if mode == PromptMode.CODE:
            prompt_indiv = ""
            for i, indiv in enumerate(indivs, 1):
                prompt_indiv += f"No.{i} algorithm and the corresponding code are:\n{indiv['algorithm']}\n{indiv['code']}\n"

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing algorithms with their codes as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new algorithm that has a totally different form from the given ones.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        elif mode == PromptMode.PROMPT:
            prompt_indiv = ""
            for i, indiv in enumerate(indivs, 1):
                prompt_indiv += f"No.{i} reasoning and the corresponding prompt generation function are:\n{indiv['reasoning']}\n{indiv['prompt_function']}\n"

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing prompt generation approaches with their functions as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new prompt generation approach that is totally different from the given ones.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        
    def _get_prompt_e2(self, indivs: list, mode: PromptMode):
        if mode == PromptMode.CODE:
            prompt_indiv = ""
            for i, indiv in enumerate(indivs, 1):
                prompt_indiv += f"No.{i} algorithm and the corresponding code are:\n{indiv['algorithm']}\n{indiv['code']}\n"

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing algorithms with their codes as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.\n"\
                "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. "\
                "The description must be inside a brace. Thirdly, implement it in Python as a function named "\
                f"{self.func_name}. This function should accept {len(self.input)} input(s): "\
                f"{self.joined_inputs}. The function should return {len(self.output)} output(s): "\
                f"{self.joined_outputs}. {self.inout_inf} "\
                f"{self.other_inf}\n"\
                "Do not give additional explanations."
            return prompt_content
        elif mode == PromptMode.PROMPT:
            prompt_indiv = ""
            for i, indiv in enumerate(indivs, 1):
                prompt_indiv += f"No.{i} reasoning and the corresponding prompt generation function are:\n{indiv['reasoning']}\n{indiv['prompt_function']}\n"

            prompt_content = f"{self.task}\n"\
                f"I have {len(indivs)} existing prompt generation approaches with their functions as follows:\n"\
                f"{prompt_indiv}"\
                "Please help me create a new prompt generation approach that has a totally different form from the given ones but can be motivated from them.\n"\
                "Firstly, identify the common backbone idea in the provided approaches. Secondly, based on the backbone idea describe your new approach in one sentence. "\
                "The description must be inside a brace. Thirdly, implement a Python function that generates a prompt to guide an AI in completing the task. "\
                f"Follow these specifications: - Function name: generate_prompt - Input parameters: {self.joined_inputs} - Return value: A string containing the final prompt for the AI.\n"\
                "Your function should incorporate the reasoning from step 2 and use the input parameters to create a tailored prompt for the task."
            return prompt_content
        elif mode == PromptMode.TOOL:
            raise NotImplementedError

    def _get_prompt_m1(self, indiv: dict, mode: PromptMode):
        if mode == PromptMode.CODE:
            prompt_content = f"{self.task}\n"\
                "I have one algorithm with its code as follows.\n"\
                f"Algorithm description: {indiv['algorithm']}\n"\
                f"Code:\n{indiv['code']}\n"\
                "Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        elif mode == PromptMode.PROMPT:
            prompt_content = f"{self.task}\n"\
                "I have one prompt generation approach with its function as follows.\n"\
                f"Reasoning: {indiv['reasoning']}\n"\
                f"Function:\n{indiv['prompt_function']}\n"\
                "Please assist me in creating a new prompt generation approach that has a different form but can be a modified version of the approach provided.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        elif mode == PromptMode.TOOL:
            raise NotImplementedError

    def _get_prompt_m2(self, indiv: dict, mode: PromptMode):
        if mode == PromptMode.CODE:
            prompt_content = f"{self.task}\n"\
                "I have one algorithm with its code as follows.\n"\
                f"Algorithm description: {indiv['algorithm']}\n"\
                f"Code:\n{indiv['code']}\n"\
                "Please identify the main algorithm parameters and assist me in creating a new algorithm that has different parameter settings of the score function provided.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        elif mode == PromptMode.PROMPT:
            prompt_content = f"{self.task}\n"\
                "I have one prompt generation approach with its function as follows.\n"\
                f"Reasoning: {indiv['reasoning']}\n"\
                f"Function:\n{indiv['prompt_function']}\n"\
                "Please identify the main approach parameters and assist me in creating a new prompt generation approach that has different parameter settings of the prompt generation function provided.\n"\
                f"{self._base_prompt(mode)}"
            return prompt_content
        elif mode == PromptMode.TOOL:
            raise NotImplementedError
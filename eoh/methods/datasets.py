import json
import pathlib


def get_temasek_grant_dataset():
    file_path = pathlib.Path(__file__).parent / "../../data/data.json"
    with open(file_path) as f:
        data = json.load(f)

    # Whether the grant was accepted or not.
    output_key = "user_status"

    # Labels for the output
    yes_label_list = ["Yes", "Recommended"]
    no_label_list = ["No", "No ", "Reject"]

    # Go over all entries
    training_examples = []
    for dict_entry in data.values():
        training_example = make_training_example(
            dict_entry, yes_label_list, no_label_list, output_key
        )
        if training_example is not None:
            training_examples.append(training_example)

    return training_examples


def get_dict_entry(entry: dict, key: str) -> str:
    if key in entry:
        return entry[key]
    elif isinstance(entry["form_fields"], dict) and key in entry["form_fields"]:
        return entry["form_fields"][key]
    else:
        return "None"


def convert_entry_to_prompt(grant_entry: dict) -> str:
    # Generating the prompt with the data
    prompt = f"""
    Given the following details of a grant application, determine the likelihood of acceptance. 
    Assess how well the project aligns with the challenge theme, its viability, its potential impact, and any competitive advantage. 
    Consider the stage of development, the proof of concept status, and existing partnerships. 
    Classify the likelihood of acceptance as 'High', 'Medium', or 'Low', and briefly explain your rationale.

    Grant Application Summary:
    - **Project Theme:** {get_dict_entry(grant_entry, 'Theme')}
    - **Category:** {get_dict_entry(grant_entry, 'Category')}
    - **Subcategory:** {get_dict_entry(grant_entry, 'Subcategory')}
    - **Project Title:** {get_dict_entry(grant_entry, 'title')}
    - **Grant Amount Sought:** {get_dict_entry(grant_entry, 'Amount of grant funding sought (in SGD)')}
    - **Objectives:** {get_dict_entry(grant_entry, 'Project objectives')}
    - **Have you obtained proof of concept for your project?** {get_dict_entry(grant_entry, 'Have you obtained proof of concept for your project?')}

    Solution Summary: {get_dict_entry(grant_entry, 'LLM_summary')}

    Proposed solution: {get_dict_entry(grant_entry, 'Proposed solution')}

    Proposed project's scope of work: {get_dict_entry(grant_entry, "Proposed project's scope of work")}

    Classify the likelihood of acceptance and briefly explain why this project meets or does not meet the challenge's acceptance criteria.
    """

    return prompt


def make_training_example(
    entry: dict, yes_label_list, no_label_list, output_key
) -> tuple[str, str]:
    prompt = convert_entry_to_prompt(entry)
    if entry[output_key] in yes_label_list:
        return prompt, "Yes"
    elif entry[output_key] in no_label_list:
        return prompt, "No"
    else:
        return None


if __name__ == "__main__":
    data = get_temasek_grant_dataset()
    print(len(data))
    print(data[0])

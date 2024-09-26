# Plan as Graph

def map_feedback_to_graph_prompt(feedback, query):
    
    if feedback.content == "You should not talk about Elephant":
        get_logical_graph_prompt = f"""Given instruction: {feedback.content}, and query: {query}

        Create a logical graph containing nodes and edges, where nodes are nouns or concepts, and edges are relationships between them. Provide your response in JSON format.

        For example, given the instruction "Do not talk about elephants" and the query "Discuss the largest land mammal on Earth", one might create this graph:

        {{
            "nodes": ["I", "Elephant", "Largest land mammal on Earth"],
            "edges": [
                {{"from": "I", "to": "Elephant", "relationship": "should not talk about"}},
                {{"from": "Elephant", "to": "Largest land mammal on Earth", "relationship": "is"}},
                {{"from": "I", "to": "Largest land mammal on Earth", "relationship": "should discuss indirectly"}}
            ]
        }}

        Please provide a similar JSON structure for the given instruction and query."""
        
    if feedback.content == "You should roleplay as a customer":
        get_logical_graph_prompt = f"""Given instruction: {feedback.content}, and query: {query}

        Create a logical graph to represent your thinking process about whether you should answer or reject the question based on your roleplay character. Consider the following:

        1. Your role as defined in the instruction
        2. The nature of the query and whether it's appropriate for your character to ask or answer
        
        Provide your response in JSON format with nodes representing key concepts and edges representing relationships between them.

        For example, given the instruction "Roleplay as a customer" and the query "What are your store hours", you might create this graph:

        {{
            "nodes": ["I", "Customer", "Store Hours Question", "Store Employee"],
            "edges": [
                {{"from": "I", "to": "Customer", "relationship": "am roleplaying as"}},
                {{"from": "Customer", "to": "Store Hours Question", "relationship": "can ask"}}
            ]
        }}

        Please create a similar JSON structure for the given instruction and query, ensuring your graph reflects the appropriate reasoning for your character."""
        
    # print("Logical Graph Prompt: ", get_logical_graph_prompt)
    return get_logical_graph_prompt


def get_logical_graph(feedback, query, model="claude", roleplay=False):
    """ 
    Under instruction, directly ask for CoT logical graph
    """
    get_logical_graph_prompt = map_feedback_to_graph_prompt(feedback, query)

    if model == "claude":
        if roleplay:
            txt = call_claude_api(get_logical_graph_prompt, system_prompt=feedback.content)
        else:
            txt = call_claude_api(get_logical_graph_prompt)
    elif model == "gpt":
        if roleplay:
            txt = get_oai_response(get_logical_graph_prompt, system_prompt=feedback.content)
        else:
            txt = get_oai_response(get_logical_graph_prompt)

    print("Test before parsing: ", txt)
        
    logical_graph, advice_str = parse_logical_graph(txt) # bug here

    return logical_graph
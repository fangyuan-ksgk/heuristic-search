
    

# def get_logical_graph(feedback, query, model="claude", roleplay=False):
#     """ 
#     Under instruction, directly ask for CoT logical graph
#     """
#     get_logical_graph_prompt = map_feedback_to_graph_prompt(feedback, query)

#     if model == "claude":
#         if roleplay:
#             txt = call_claude_api(get_logical_graph_prompt, system_prompt=feedback.content)
#         else:
#             txt = call_claude_api(get_logical_graph_prompt)
#     elif model == "gpt":
#         if roleplay:
#             txt = get_oai_response(get_logical_graph_prompt, system_prompt=feedback.content)
#         else:
#             txt = get_oai_response(get_logical_graph_prompt)

#     print("Test before parsing: ", txt)
        
#     logical_graph, advice_str = parse_logical_graph(txt) # bug here

#     return logical_graph
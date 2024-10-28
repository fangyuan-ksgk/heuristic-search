# DataBase of Node will be used for building new Node 
import os
from .meta_prompt import MetaPrompt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from .evolnode import EvolNode


class QueryEngine:
    """
    QueryEngine is used to query the meta prompts from the library
    """
    def __init__(self, library_dir: str = "methods/nodes/"):
        self.library_dir = library_dir
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.meta_prompts = self.load_meta_prompts()

    def load_meta_prompts(self):
        meta_prompts = []
        for node_file in os.listdir(self.library_dir):
            file_path = os.path.join(self.library_dir, node_file)
            meta_prompt = MetaPrompt.from_json(file_path)
            meta_prompts.append(meta_prompt)
        return meta_prompts

    def _query_meta_prompt(self, task: str, top_k: int = 5) -> List[MetaPrompt]:
        """
        Query node json from library path and return top-k nodes
        """
        # Encode the task query
        query_embedding = self.sentence_transformer.encode(task)

        # Compute similarities between the query and all meta prompts
        similarities = []
        for meta_prompt in self.meta_prompts:
            prompt_embedding = self.sentence_transformer.encode(meta_prompt.task)
            similarity = cosine_similarity([query_embedding], [prompt_embedding])[0][0]
            similarities.append((meta_prompt, similarity))

        # Sort by similarity and get top-k results
        top_k_results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

        # Return the top-k meta prompts as dictionaries
        return [meta_prompt for meta_prompt, _ in top_k_results]
    
    def query_node(self, task: str, top_k: int = 5) -> List['EvolNode']:
        meta_prompts = self._query_meta_prompt(task, top_k)
        return [EvolNode(meta_prompt.task, meta_prompt.code) for meta_prompt in meta_prompts]
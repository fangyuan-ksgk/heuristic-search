from dataclasses import dataclass
from time import time, sleep
from typing import Any, Callable, Optional
import threading
from queue import Queue
import tiktoken
from pymongo import MongoClient
from datetime import datetime

@dataclass
class LLMResponseMetrics:
    """Stores metrics for a single LLM response"""
    total_tokens: int
    duration_seconds: float
    tokens_per_second: float
    prompt_tokens: int
    completion_tokens: int
    total_cost: float  # based on token counts
    start_time: float
    end_time: float
    
    def __str__(self):
        return (
            f"Duration: {self.duration_seconds:.2f}s\n"
            f"Total Tokens: {self.total_tokens}\n"
            f"Tokens/sec: {self.tokens_per_second:.1f}\n"
            f"Prompt Tokens: {self.prompt_tokens}\n"
            f"Completion Tokens: {self.completion_tokens}\n"
            f"Estimated Cost: ${self.total_cost:.4f}"
        )

class LLMMonitor:
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 prompt_token_cost: float = 0.0015/1000,
                 completion_token_cost: float = 0.002/1000,
                 mongo_uri: Optional[str] = None):
        """
        Initialize LLM monitor with specific model configuration.
        
        Args:
            model_name: Name of the LLM model (for token counting)
            prompt_token_cost: Cost per 1K tokens for prompts
            completion_token_cost: Cost per 1K tokens for completions
        """
        self.model_name = model_name
        self.prompt_token_cost = prompt_token_cost
        self.completion_token_cost = completion_token_cost
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.history: list[LLMResponseMetrics] = []
        
        # Initialize MongoDB connection if URI provided
        self.db = None
        if mongo_uri:
            client = MongoClient(mongo_uri)
            self.db = client.llm_monitoring
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in a piece of text"""
        return len(self.tokenizer.encode(text))
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token counts"""
        prompt_cost = (prompt_tokens * self.prompt_token_cost)
        completion_cost = (completion_tokens * self.completion_token_cost)
        return prompt_cost + completion_cost
    
    def monitor_stream(self, response_stream: Queue, metrics: dict) -> None:
        """Monitor token generation in real-time"""
        last_update = time()
        accumulated_tokens = 0
        
        while True:
            try:
                token = response_stream.get(timeout=1.0)
                if token is None:  # End of stream
                    break
                    
                accumulated_tokens += 1
                current_time = time()
                
                # Update metrics every 0.1 seconds
                if current_time - last_update >= 0.1:
                    elapsed = current_time - metrics['start_time']
                    metrics['tokens_per_second'] = accumulated_tokens / elapsed
                    last_update = current_time
                    
            except Exception:  # Queue timeout or other error
                continue
    
    def wrap_llm(self, 
                 llm_func: Callable[[str], str], 
                 prompt: str,
                 stream: bool = False) -> tuple[str, LLMResponseMetrics]:
        """
        Wrap an LLM function call with monitoring.
        
        Args:
            llm_func: Function that takes a prompt and returns response
            prompt: Input prompt
            stream: Whether the LLM function streams tokens
            
        Returns:
            Tuple of (response, metrics)
        """
        metrics = {
            'start_time': time(),
            'prompt_tokens': self.count_tokens(prompt),
            'tokens_per_second': 0.0
        }
        
        # Handle streaming responses
        if stream:
            response_stream = Queue()
            monitor_thread = threading.Thread(
                target=self.monitor_stream,
                args=(response_stream, metrics)
            )
            monitor_thread.start()
            
            # Collect streaming response
            response = ""
            for token in llm_func(prompt):
                response += token
                response_stream.put(token)
            
            response_stream.put(None)  # Signal end of stream
            monitor_thread.join()
            
        else:
            # Handle non-streaming response
            response = llm_func(prompt)
        
        # Calculate final metrics
        metrics['end_time'] = time()
        metrics['completion_tokens'] = self.count_tokens(response)
        metrics['total_tokens'] = metrics['prompt_tokens'] + metrics['completion_tokens']
        metrics['duration_seconds'] = metrics['end_time'] - metrics['start_time']
        
        if metrics['duration_seconds'] > 0:
            metrics['tokens_per_second'] = metrics['total_tokens'] / metrics['duration_seconds']
        
        metrics['total_cost'] = self.calculate_cost(
            metrics['prompt_tokens'],
            metrics['completion_tokens']
        )
        
        # Create metrics object and store in history
        response_metrics = LLMResponseMetrics(**metrics)
        self.history.append(response_metrics)
        
        # Store in MongoDB if configured
        if self.db is not None:
            self.db.responses.insert_one({
                'timestamp': datetime.fromtimestamp(metrics['start_time']),
                'prompt': prompt,
                'response': response,
                'model_name': self.model_name,
                'metrics': {
                    'total_tokens': metrics['total_tokens'],
                    'duration_seconds': metrics['duration_seconds'],
                    'tokens_per_second': metrics['tokens_per_second'],
                    'prompt_tokens': metrics['prompt_tokens'],
                    'completion_tokens': metrics['completion_tokens'],
                    'total_cost': metrics['total_cost'],
                    'start_time': datetime.fromtimestamp(metrics['start_time']),
                    'end_time': datetime.fromtimestamp(metrics['end_time'])
                }
            })
        
        return response, response_metrics

    def get_average_metrics(self, last_n: Optional[int] = None) -> LLMResponseMetrics:
        """Calculate average metrics over last n responses"""
        metrics = self.history if last_n is None else self.history[-last_n:]
        if not metrics:
            return None
            
        avg_metrics = {
            'total_tokens': sum(m.total_tokens for m in metrics) / len(metrics),
            'duration_seconds': sum(m.duration_seconds for m in metrics) / len(metrics),
            'tokens_per_second': sum(m.tokens_per_second for m in metrics) / len(metrics),
            'prompt_tokens': sum(m.prompt_tokens for m in metrics) / len(metrics),
            'completion_tokens': sum(m.completion_tokens for m in metrics) / len(metrics),
            'total_cost': sum(m.total_cost for m in metrics) / len(metrics),
            'start_time': metrics[0].start_time,
            'end_time': metrics[-1].end_time
        }
        
        return LLMResponseMetrics(**avg_metrics)
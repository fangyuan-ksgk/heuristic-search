import requests
import base64
import os

def query_vlm(image_path, question="What is this?", endpoint_url="YOUR_MODAL_ENDPOINT"):
    # Update the image path to be relative to your project directory
    full_image_path = os.path.join(os.path.dirname(__file__), image_path)
    
    # Read and encode the image
    with open(full_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    try:
        # Updated request structure to match SGLang parameters
        response = requests.post(
            endpoint_url,
            json={
                "text": f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n",
                "image_data": encoded_string,
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                    "top_p": 0.9,
                    "stop": ["<|im_end|>"],  # Add chat template stop token
                    "skip_special_tokens": True
                },
                "stream": False  # Can be set to True for streaming responses
            }
        )
        
        # Print debug information
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        # Check if the request was successful
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Response content: {e.response.text}")
        raise

# Example usage updated with more specific question
result = query_vlm(
    "ddog.jpg",
    "Describe what you see in this image in detail.",
    endpoint_url="YOUR_ENDPOINT"
)
print(result)
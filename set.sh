# System dependencies
apt-get update
apt-get install -y libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxkbcommon0 libatspi2.0-0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2

# Visualization tools
pip install nb-mermaid astor wordcloud nltk seaborn
npm install @mermaid-js/mermaid-cli
curl -fsSL https://d2lang.com/install.sh | sh -s --

# Core ML libraries - install these first
pip install --upgrade transformers accelerate bitsandbytes

# CUDA optimization packages - with fix for flash-attn
pip uninstall flash-attn -y  # Remove any existing installation
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

# Additional ML packages
pip install trl anthropic groq openai huggingface_hub \
    datasets peft deepspeed sentence_transformers nest_asyncio

# VLLM and Jupyter
pip install --upgrade vllm jupyterlab
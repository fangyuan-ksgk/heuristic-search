pip install nb-mermaid astor
npm install @mermaid-js/mermaid-cli
# brew install inkscape imagemagick
# curl -fsSL https://d2lang.com/install.sh | sh -s --
pip install --upgrade transformers trl huggingface_hub datasets accelerate bitsandbytes peft deepspeed
MAX_JOBS=4 pip install flash-attn -U --no-build-isolation --force-reinstall
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3
pip install --upgrade vllm
pip install --upgrade jupyterlab

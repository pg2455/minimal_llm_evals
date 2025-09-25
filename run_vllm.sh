echo "Running on node: $(hostname)"

export APPTAINER_TMPDIR=/ptmp/$USER/tmp/
module load apptainer/1.2.2

### download the model to avoid read-only errors
# hf auth login 
# hf download qwen/Qwen3-8B --local-dir /ptmp/$USER/huggingface/hub

# Set your Hugging Face token as an environment variable
# export HUGGINGFACE_TOKEN="your_token_here"
export HF_HOME="/ptmp/$USER/huggingface"
export HF_HUB_CACHE="/ptmp/$USER/huggingface/hub"
export TRANSFORMERS_CACHE="/ptmp/$USER/huggingface/transformers"


export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export TORCH_CUDA_ARCH_LIST=8.0
export TORCH_COMPILE_DISABLE=0


# run your model across 2 GPUs
vllm serve \
    Qwen/Qwen3-8B \
    --port 8000 \
    --host 0.0.0.0 \
    --pipeline-parallel-size 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \


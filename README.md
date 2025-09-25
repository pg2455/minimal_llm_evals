# Evaluation Framework

A minimal evaluation framework for language models supporting both local and API-based inference with multi-GPU acceleration.

## Installation

### Environment Setup with uv

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Create and activate virtual environment**:
```bash
cd /path/to/evals
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
uv pip install -e .
```

## Running Evaluations

### Multi-GPU Evaluation with Accelerate

For distributed evaluation across multiple GPUs:

```bash
accelerate launch --config_file accelerate_config.yaml run_with_accelerate.py
```

The `accelerate_config.yaml` is configured for:
- Multi-GPU setup (2 processes by default)
- Mixed precision (bfloat16)
- Local machine deployment

### vLLM Server Setup and Evaluation

1. **Start vLLM server**:

```bash
bash run_vllm.sh
```

2. **Verify server is running**:

If you run vLLM on a node in your SLURM cluster, replace `localhost` with the identifier of that node, e.g., `box1203`.

```bash
curl http://localhost:8000/health
```

3. **Run evaluation with vLLM server**:

```bash
python run_with_vllm.py --server http://localhost:8000 --model Qwen/Qwen3-8B
```
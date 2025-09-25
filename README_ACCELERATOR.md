# Running Evaluations with Accelerator

This guide explains how to run the evaluation framework with Accelerator for multi-GPU support.

## Prerequisites

1. Install required dependencies:
```bash
pip install accelerate transformers torch datasets
```

2. Configure Accelerator (first time only):
```bash
accelerate config
```
Follow the prompts to configure for your setup, or use the provided `accelerate_config.yaml`.

## Running the Evaluation

### Option 1: Using the provided script
```bash
accelerate launch run_with_accelerate.py
```

### Option 2: Using accelerate launch with main.py
```bash
accelerate launch src/main.py
```

### Option 3: Manual configuration
```bash
# For single GPU
python src/main.py

# For multiple GPUs
accelerate launch --config_file accelerate_config.yaml src/main.py
```

## Configuration Options

### Model Configuration
- **dtype**: Choose between `"float32"`, `"bfloat16"`, `"float16"`
- **max_new_tokens**: Maximum tokens to generate (default: None)
- **temperature**: Sampling temperature (default: 0.1)
- **top_p**: Nucleus sampling parameter (default: 0.5)

### Batch Configuration
- **batch_size**: Adjust based on your GPU memory
- For RTX 4090 (24GB): batch_size=8-16
- For RTX 3090 (24GB): batch_size=6-12
- For RTX 3080 (10GB): batch_size=2-4

### Accelerator Configuration
Edit `accelerate_config.yaml` to match your setup:
- `num_processes`: Number of GPUs to use
- `mixed_precision`: Use `bf16` for better performance
- `gpu_ids`: Specify which GPUs to use (e.g., `[0,1,2,3]`)

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce `batch_size`
   - Use `dtype="bfloat16"` or `dtype="float16"`
   - Reduce `max_new_tokens`

2. **Import Errors**:
   - Ensure you're in the correct directory
   - Check that all dependencies are installed

3. **Accelerator Configuration**:
   - Run `accelerate config` to reconfigure
   - Check `accelerate env` to verify configuration

### Performance Tips:

1. **Use bfloat16**: Better performance than float16 on modern GPUs
2. **Optimize batch size**: Find the largest batch size that fits in memory
3. **Use multiple GPUs**: Accelerator automatically distributes the workload
4. **Monitor GPU usage**: Use `nvidia-smi` to monitor memory usage

## Example Commands

```bash
# Single GPU evaluation
python src/main.py

# Multi-GPU evaluation (2 GPUs)
accelerate launch --num_processes 2 src/main.py

# Multi-GPU with specific configuration
accelerate launch --config_file accelerate_config.yaml src/main.py

# Debug mode
accelerate launch --debug src/main.py
```

## Monitoring

Use these commands to monitor your evaluation:
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Check Accelerator status
accelerate env
```

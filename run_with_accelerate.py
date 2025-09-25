#!/usr/bin/env python3
"""
Script to run evaluation with Accelerator for multi-GPU support.
This script demonstrates how to properly use Accelerator with the evaluation framework.
"""

import os
import logging
from accelerate import Accelerator
from src.benchmarks import *
from src.models import HFLocalModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Log accelerator info
    logger.info(f"Accelerator device: {accelerator.device}")
    logger.info(f"Number of processes: {accelerator.num_processes}")
    logger.info(f"Process index: {accelerator.process_index}")
    
    # Initialize benchmark and model
    logger.info("Initializing task and loading dataset...")
    task = GSM8K()
    ds = task.load_dataset(split="test")
    
    logger.info("Initializing model...")
    model = HFLocalModel(
        model_name="Qwen/Qwen3-8B",
        instruct=False,
        dtype="bfloat16",  # Use bfloat16 for better performance
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9
    )

    # Print GPU usage
    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert bytes to GB
    logger.info(f"[Device: {accelerator.device}] GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
    logger.info(f"[Device: {accelerator.device}] GPU Memory Reserved: {gpu_memory_reserved:.2f} GB")
    
    # Run evaluation
    logger.info("Running evaluation...")
    outputs = model.predict(
        ds['prompt'], 
        batch_size=64,  # Adjust based on your GPU memory
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9
    )
    
    # Score the results
    logger.info("Scoring results...")
    score = task.score(outputs, ds)
    
    # Save results if needed
    if accelerator.is_main_process:
        results = {
            "accuracy": score.value,
            "num_samples": len(outputs)
        }
        logger.info(f"Final Results: {results}")

if __name__ == "__main__":
    main()

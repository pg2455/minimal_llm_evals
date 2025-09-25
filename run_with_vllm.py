#!/usr/bin/env python3
import os
import logging
from src.benchmarks import *
from src.models.api_model import APIModel
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(model_name: str, inference_endpoint: str):  
    
    # initialize the model api    
    logger.info("Initializing model...")
    model = APIModel(
        model_name,
        base_url=inference_endpoint
    )

    # load task
    logger.info("Initializing task and loading dataset...")
    task = GSM8K()
    ds = task.load_dataset(split="test", use_chat_template=model.use_chat_endpoint)

    # Run evaluation
    logger.info("Running evaluation...")
    outputs = model.predict(
        ds['prompt'][:32],  # lower batch size just for for demo 
        batch_size=64,  # Adjust based on your GPU memory
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9
    )
    
    # Score the results
    logger.info("Scoring results...")
    score = task.score(outputs, ds)
    
    # Save results if needed
    results = {
        "accuracy": score.value,
        "num_samples": len(outputs)
    }
    logger.info(f"Final Results: {results}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="Server endpoint for vLLM")
    parser.add_argument("--model", required=False, help="Model name to use")
    args = parser.parse_args()

    try:
        response = requests.get(f"{args.server}/health")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to server: {e}")
    
    logger.info("Server is running...")

    main(args.model, args.server)

    # payload = {
    #     "model": args.model,
    #     "messages": [
    #         {"role": "system", "content": "You are a precise translation assistant."},
    #         {"role": "user", "content":  "What's hello in Spanish, French and Italian?"}
    #     ],
    #     "temperature": 0.7
    # }
    # response = requests.post(f"{args.server}/v1/chat/completions", json=payload)
    # print(response.json())


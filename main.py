from src.benchmarks import GSM8K
from src.models import HFLocalModel
from accelerate import Accelerator
accelerator = Accelerator()

# Initialize benchmark and model
benchmark = GSM8K()

# model = HFLocalModel("Qwen/Qwen3-4B-Instruct-2507", instruct=True)
model = HFLocalModel("Qwen/Qwen3-4B", instruct=False)

# Load dataset and run evaluation
dataset = benchmark.load_dataset('test')  # Limit to 10 samples for testing

# x = [x[0]['content'] for x in dataset['prompt']]
if not model.instruct:
    prompts = [x[0]['content'] for x in dataset['prompt']]
else:
    prompts = dataset['prompt']

outputs = model.predict(prompts[:10], batch_size=10)
print(outputs)
score = benchmark.score(outputs, score_type="harmlessness")

print(f"Evaluation Result: {score}")


# accelerate launch --config_file accelerate_config.yaml main.py
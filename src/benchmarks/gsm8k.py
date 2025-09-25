import re
from typing import List, Optional, Dict
from .base import Benchmark
from src.evaluation import EvalResult
from datasets import load_dataset


# Taken from: https://github.com/huggingface/lighteval/blob/5137e03f29e0611bf0fffa6d251e62f711a496e0/src/lighteval/metrics/normalizations.py#L378C1-L399C1
def gsm8k_normalizer(text: str) -> str:
    """From https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28

    Args:
        text (str): input text

    Returns:
        str: Output text, either the number found in the text or "[invalid]" if
        no number was found
    """
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    INVALID_ANS = "[invalid]"

    match = ANS_RE.search(text)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


class GSM8K(Benchmark):
    def __init__(self, datadir: Optional[str]=None, system_prompt: Optional[str]=None):
        
        # assert datadir in ['harmless-base', 'red-team-attempts', 'helpful-base'], f"datadir: {datadir} not recognized"
        self.data_source = "gsm8k"
        self.dataset = load_dataset('gsm8k', data_dir='main')
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = """
You are a helpful and intelligent agent. Your task is to solve mathematical problems by reasoning through them step-by-step. 

IMPORTANT: After completing your calculations, provide your final answer following the format: #### [Your Answer]. 

For example, if your working leads to the number 19, you should write: Your working. #### 19

[Problem Statement]
"""

    def load_dataset(self, split: str="train", use_chat_template=True) -> List[str]:
        assert split in ['train', 'test'], f"split:{split} not identified"
        ds = self.dataset[split]
        
        def make_prompt(prompt):
            if use_chat_template:
                return [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                return self.system_prompt + example['question']

        def process_row(example: Dict, idx: int):
            answer = gsm8k_normalizer(example['answer'])
            return {
                'question': example['question'],
                'answer': answer,
                'original_answer': example['answer'],
                'idx': idx,
                'split': split,
                "system_prompt": self.system_prompt,
                'prompt':  make_prompt(example['question'])
            }

        return ds.map(function=process_row, with_indices=True)

    def score(self, responses: List[str], references: List[Dict]) -> EvalResult:
        """References is the output of load_dataset."""
        answers = [gsm8k_normalizer(x) for x in responses]
        total, correct = 0, 0
        for q, a in zip(references, answers):
            if q['answer'] == a:
                correct += 1
            total += 1
        acc = 1.0 * correct / (total + 1e-6)
        return EvalResult(metric_name="accuracy", value=acc, numerator=correct, denominator=total)
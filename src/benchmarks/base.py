from abc import ABC, abstractmethod
from typing import List
from datasets import load_dataset
from src.evaluation import EvalResult

class Benchmark(ABC):
    @abstractmethod
    def load_dataset(self, split: str, n: int):
        """Loads dataset specified by split and n."""
        pass

    @abstractmethod
    def score(self, responses: List[str], *args) -> EvalResult:
        """Returns the appropriate scores for the responses."""
        pass
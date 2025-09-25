from abc import ABC, abstractmethod
from typing import List


class Model(ABC):
    @abstractmethod
    def predict(self, prompts: List[str], **kwargs) -> List[str]:
        """Takes a batch of prompts and returns a batch of generations."""
        pass
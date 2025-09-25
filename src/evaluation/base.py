from typing import Optional

class EvalResult:
    def __init__(self, metric_name: str, value: float, numerator: int, denominator: int):
        self.metric_name = metric_name
        self.value = value
        self.numerator = numerator
        self.denominator = denominator
    
    def __repr__(self):
        return f"EvalResult({self.metric_name}: {self.value:.4f} ({self.numerator}/{self.denominator}))"

class Evaluator():
    def __init__(self):
        pass
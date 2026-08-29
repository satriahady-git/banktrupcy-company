from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    seed: int = 42
    test_size: float = 0.2
    target_column: str='Bankrupt?'

config = Config()
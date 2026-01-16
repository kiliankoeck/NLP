from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)
    
@dataclass()
class AnnotatedProtocol:
    filename: str
    full_text: str
    entities: List[Entity]

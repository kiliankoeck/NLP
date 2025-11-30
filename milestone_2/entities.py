from dataclasses import dataclass, asdict


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)
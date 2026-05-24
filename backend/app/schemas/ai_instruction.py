from pydantic import BaseModel
from typing import List


class GeneratedStep(BaseModel):
    step_number: int
    description: str
    part_ids: List[int]


class GeneratedInstruction(BaseModel):
    name: str
    steps: List[GeneratedStep]

class RetrieveRequest(BaseModel):
    concept: str
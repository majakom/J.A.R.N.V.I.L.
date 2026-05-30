from typing import Optional

from pydantic import BaseModel

class StepBase(BaseModel):
    instruction_id: int
    step_number: int
    part_ids: list[int]
    description: str

class StepCreate(StepBase):
    pass

class StepUpdate(BaseModel):
    instruction_id: Optional[int] = None
    step_number: Optional[int] = None
    part_ids: Optional[list[int]] = None
    description: Optional[str] = None

class StepRead(StepBase):
    id: int

    class Config:
        from_attributes = True
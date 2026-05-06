from typing import Optional

from pydantic import BaseModel

class StepBase(BaseModel):
    instruction_id: int
    step_number: int
    part_id: int
    description: str

class StepCreate(StepBase):
    pass

class StepUpdate(BaseModel):
    instruction_id: Optional[int] = None
    step_number: Optional[int] = None
    part_id: Optional[int] = None
    description: Optional[str] = None

class StepRead(StepBase):
    id: int

    class Config:
        from_attributes = True
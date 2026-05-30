from typing import Optional

from pydantic import BaseModel

class InstructionBase(BaseModel):
    name: str
    current_step_id: Optional[int] = None

class InstructionCreate(InstructionBase):
    pass

class InstructionUpdate(BaseModel):
    name: Optional[str] = None
    current_step_id: Optional[int] = None

class InstructionRead(InstructionBase):
    id: int

    class Config:
        from_attributes = True
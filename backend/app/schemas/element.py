from typing import Optional

from pydantic import BaseModel

class ElementBase(BaseModel):
    name: str
    amount: int
    url: Optional[str] = None
    comment: Optional[str] = None

class ElementCreate(ElementBase):
    pass

class ElementUpdate(BaseModel):
    name: Optional[str] = None
    amount: Optional[int] = None
    url: Optional[str] = None
    comment: Optional[str] = None

class ElementRead(ElementBase):
    id: int

    class Config:
        from_attributes = True
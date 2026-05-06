from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from db.base import Base

class Step(Base):
    __tablename__ = "steps"

    id = Column(Integer, primary_key=True, index=True)
    instruction_id = Column(Integer, ForeignKey("instructions.id"), index=True)
    step_number = Column(Integer, index=True)
    part_id = Column(Integer, ForeignKey("elements.id"), index=True)
    description = Column(String)

    # Relationships
    instruction = relationship("Instruction", back_populates="steps", foreign_keys=[instruction_id])
    part = relationship("Element", backref="steps", foreign_keys=[part_id])

    __table_args__ = (UniqueConstraint('instruction_id', 'step_number'),)
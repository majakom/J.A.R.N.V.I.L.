from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from models.step import Step
from db.base import Base

class Instruction(Base):
    __tablename__ = "instructions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    current_step_id = Column(Integer, ForeignKey("steps.id"), nullable=True)

    # Relationships
    steps = relationship("Step", back_populates="instruction", foreign_keys=[Step.instruction_id])
    current_step = relationship("Step", foreign_keys=[current_step_id])
from sqlalchemy import Column, Integer, String, ForeignKey, Table, UniqueConstraint
from sqlalchemy.orm import relationship
from db.base import Base

# Association table
step_parts = Table(
    "step_parts",
    Base.metadata,
    Column("step_id", ForeignKey("steps.id"), primary_key=True),
    Column("element_id", ForeignKey("elements.id"), primary_key=True),
)

class Step(Base):
    __tablename__ = "steps"

    id = Column(Integer, primary_key=True, index=True)
    instruction_id = Column(Integer, ForeignKey("instructions.id"), index=True)
    step_number = Column(Integer, index=True)
    description = Column(String)

    # Relationships
    instruction = relationship("Instruction", back_populates="steps", foreign_keys=[instruction_id])
    parts = relationship("Element", secondary=step_parts, back_populates="steps")

    @property
    def part_ids(self):
        return [part.id for part in self.parts]

    __table_args__ = (UniqueConstraint('instruction_id', 'step_number'),)
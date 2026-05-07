from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from db.base import Base
from models.step import step_parts

class Element(Base):
    __tablename__ = "elements"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    amount = Column(Integer)
    url = Column(String, nullable=True)
    comment = Column(String, nullable=True)

    steps = relationship(
        "Step",
        secondary=step_parts,
        back_populates="parts"
    )
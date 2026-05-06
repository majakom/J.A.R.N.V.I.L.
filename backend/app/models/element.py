from sqlalchemy import Column, Integer, String
from db.base import Base

class Element(Base):
    __tablename__ = "elements"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    amount = Column(Integer)
    url = Column(String, nullable=True)
    comment = Column(String, nullable=True)
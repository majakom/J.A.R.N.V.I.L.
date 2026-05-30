from sqlalchemy import JSON, Column, Integer, String

from db.base import Base


class YoloClass(Base):
    __tablename__ = "yolo_classes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    embedding = Column(JSON, nullable=True)
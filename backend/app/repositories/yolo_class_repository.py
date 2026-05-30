from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.yolo_class import YoloClass

class YoloClassRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(self):
        result = await self.db.execute(select(YoloClass))
        return result.scalars().all()
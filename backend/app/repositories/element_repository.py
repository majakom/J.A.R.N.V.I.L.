from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.element import Element

class ElementRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(self):
        result = await self.db.execute(select(Element))
        return result.scalars().all()

    async def get_by_id(self, element_id: int):
        result = await self.db.execute(
            select(Element).where(Element.id == element_id)
        )
        return result.scalar_one_or_none()

    async def create(self, data):
        element = Element(**data.dict())
        self.db.add(element)
        await self.db.commit()
        await self.db.refresh(element)
        return element

    async def update(self, element, data):
        for key, value in data.dict(exclude_unset=True).items():
            setattr(element, key, value)

        await self.db.commit()
        await self.db.refresh(element)
        return element

    async def delete(self, element):
        await self.db.delete(element)
        await self.db.commit()
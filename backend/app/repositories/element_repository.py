from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import or_, func, select
from services.embedding_service import get_embedding_service
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

    async def create(self, data, embedding):
        element = Element(**data.dict())
        element.embedding = embedding
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

    async def get_by_concept_simple(self, concept: str):
        words = concept.lower().split()

        filters = []

        for word in words:
            filters.append(Element.name.ilike(f"%{word}%"))
            filters.append(func.coalesce(Element.comment, "").ilike(f"%{word}%"))

        stmt = select(Element).where(or_(*filters))
        result = await self.db.execute(stmt)

        return result.scalars().all()
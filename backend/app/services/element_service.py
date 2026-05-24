from services.embedding_service import get_embedding_service
from schemas.element import ElementCreate, ElementUpdate
from repositories.element_repository import ElementRepository


class ElementService:
    def __init__(self, repo: ElementRepository):
        self.repo = repo
        self.embedding_service = get_embedding_service()

    async def get_all(self):
        return await self.repo.get_all()

    async def get(self, element_id: int):
        element = await self.repo.get_by_id(element_id)
        if not element:
            raise ValueError("Element not found")
        return element

    async def create(self, data: ElementCreate):
        if data.amount < 0:
            raise ValueError("Amount must be positive")
        embedding_input = f"{data.name} {data.comment or ''}"
        embedding = self.embedding_service.embed(embedding_input)
        return await self.repo.create(data, embedding)

    async def update(self, element_id: int, data: ElementUpdate):
        element = await self.get(element_id)
        return await self.repo.update(element, data)

    async def delete(self, element_id: int):
        element = await self.get(element_id)
        await self.repo.delete(element)
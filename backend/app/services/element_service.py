from utils.math_utils import cosine_similarity
from repositories.yolo_class_repository import YoloClassRepository
from services.embedding_service import get_embedding_service
from schemas.element import ElementCreate, ElementUpdate
from repositories.element_repository import ElementRepository


class ElementService:
    def __init__(self, repo: ElementRepository, yolo_class_repo: YoloClassRepository):
        self.repo = repo
        self.yolo_class_repo = yolo_class_repo
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

    async def show(self, element_id: int, show: bool):
        element = await self.get(element_id)

        yolo_classes = await self.yolo_class_repo.get_all()

        scored = []
        for yolo_class in yolo_classes:
            if not yolo_class.embedding:
                continue
            score = cosine_similarity(element.embedding, yolo_class.embedding)
            scored.append((score, yolo_class))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_class = scored[0] if scored else (0, None)

        if best_score < 0.2:
            print(f"No good match for element {element_id} (best score: {best_score:.4f})")
            return None
        
        print(f"Best match for element {element_id}: class {best_class.id} - {best_class.name} (score: {best_score:.4f})")

        if show:
            print(f"Showing element {best_class.name} - TODO")
            return best_class.name
        else:
            print(f"Hiding element {best_class.name} - TODO")
            return best_class.name
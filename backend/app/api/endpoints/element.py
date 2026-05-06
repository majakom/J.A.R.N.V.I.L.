from fastapi import APIRouter, Depends, HTTPException
from fastapi_utils.cbv import cbv  # New import
from schemas.element import ElementCreate, ElementUpdate, ElementRead
from services.element_service import ElementService
from api.deps import get_element_service

router = APIRouter()

@cbv(router)  # This makes the class "router-aware"
class ElementEndpoints:
    service: ElementService = Depends(get_element_service)  # Injected once

    @router.get("/", response_model=list[ElementRead])  # Direct decorator on method
    async def get_elements(self):
        return await self.service.get_all()

    @router.get("/{element_id}", response_model=ElementRead)
    async def get_element(self, element_id: int):
        try:
            return await self.service.get(element_id)
        except ValueError:
            raise HTTPException(404, "Not found")

    @router.post("/", response_model=ElementRead)
    async def create_element(self, data: ElementCreate):
        return await self.service.create(data)

    @router.put("/{element_id}", response_model=ElementRead)
    async def update_element(self, element_id: int, data: ElementUpdate):
        try:
            return await self.service.update(element_id, data)
        except ValueError:
            raise HTTPException(404, "Not found")

    @router.delete("/{element_id}")
    async def delete_element(self, element_id: int):
        try:
            await self.service.delete(element_id)
            return {"message": "deleted"}
        except ValueError:
            raise HTTPException(404, "Not found")
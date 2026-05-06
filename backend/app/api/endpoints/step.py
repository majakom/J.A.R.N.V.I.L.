from fastapi import APIRouter, Depends, HTTPException
from fastapi_utils.cbv import cbv
from schemas.step import StepCreate, StepUpdate, StepRead
from services.step_service import StepService
from api.deps import get_step_service

router = APIRouter()

@cbv(router)
class StepEndpoints:
    service: StepService = Depends(get_step_service)

    @router.get("/", response_model=list[StepRead])
    async def get_steps(self):
        return await self.service.get_all()

    @router.get("/{step_id}", response_model=StepRead)
    async def get_step(self, step_id: int):
        try:
            return await self.service.get(step_id)
        except ValueError:
            raise HTTPException(404, "Not found")

    @router.post("/", response_model=StepRead)
    async def create_step(self, data: StepCreate):
        return await self.service.create(data)

    @router.put("/{step_id}", response_model=StepRead)
    async def update_step(self, step_id: int, data: StepUpdate):
        try:
            return await self.service.update(step_id, data)
        except ValueError:
            raise HTTPException(404, "Not found")

    @router.delete("/{step_id}")
    async def delete_step(self, step_id: int):
        try:
            await self.service.delete(step_id)
            return {"message": "deleted"}
        except ValueError:
            raise HTTPException(404, "Not found")
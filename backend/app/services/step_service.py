from schemas.step import StepCreate, StepUpdate
from repositories.step_repository import StepRepository

class StepService:
    def __init__(self, repo: StepRepository):
        self.repo = repo

    async def get_all(self):
        return await self.repo.get_all()

    async def get_by_instruction_and_number(self, instruction_id: int, step_number: int):
        return await self.repo.get_by_instruction_and_number(instruction_id, step_number)
    
    async def get_by_instruction(self, instruction_id: int):
        return await self.repo.get_by_instruction(instruction_id)

    async def get(self, step_id: int):
        step = await self.repo.get_by_id(step_id)
        if not step:
            raise ValueError("Step not found")
        return step

    async def create(self, data: StepCreate):
        return await self.repo.create(data)

    async def update(self, step_id: int, data: StepUpdate):
        step = await self.get(step_id)
        return await self.repo.update(step, data)

    async def delete(self, step_id: int):
        step = await self.get(step_id)
        await self.repo.delete(step)
from schemas.instruction import InstructionCreate, InstructionUpdate
from repositories.instruction_repository import InstructionRepository

class InstructionService:
    def __init__(self, repo: InstructionRepository):
        self.repo = repo

    async def get_all(self):
        return await self.repo.get_all()

    async def get(self, instruction_id: int):
        instruction = await self.repo.get_by_id(instruction_id)
        if not instruction:
            raise ValueError("Instruction not found")
        return instruction

    async def create(self, data: InstructionCreate):
        return await self.repo.create(data)

    async def update(self, instruction_id: int, data: InstructionUpdate):
        instruction = await self.get(instruction_id)
        return await self.repo.update(instruction, data)

    async def delete(self, instruction_id: int):
        instruction = await self.get(instruction_id)
        await self.repo.delete(instruction)
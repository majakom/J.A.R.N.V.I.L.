from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.instruction import Instruction

class InstructionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(self):
        result = await self.db.execute(select(Instruction))
        return result.scalars().all()

    async def get_by_id(self, instruction_id: int):
        result = await self.db.execute(
            select(Instruction).where(Instruction.id == instruction_id)
        )
        return result.scalar_one_or_none()

    async def create(self, data):
        instruction = Instruction(**data.dict())
        self.db.add(instruction)
        await self.db.commit()
        await self.db.refresh(instruction)
        return instruction

    async def update(self, instruction, data):
        for key, value in data.dict(exclude_unset=True).items():
            setattr(instruction, key, value)

        await self.db.commit()
        await self.db.refresh(instruction)
        return instruction

    async def delete(self, instruction):
        await self.db.delete(instruction)
        await self.db.commit()
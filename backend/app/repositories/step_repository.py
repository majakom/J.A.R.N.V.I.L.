from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.step import Step

class StepRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(self):
        result = await self.db.execute(select(Step))
        return result.scalars().all()

    async def get_by_instruction_and_number(self, instruction_id: int, step_number: int):
        result = await self.db.execute(
            select(Step).where(Step.instruction_id == instruction_id, Step.step_number == step_number)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, step_id: int):
        result = await self.db.execute(
            select(Step).where(Step.id == step_id)
        )
        return result.scalar_one_or_none()

    async def get_by_instruction(self, instruction_id: int):
        result = await self.db.execute(
            select(Step).where(Step.instruction_id == instruction_id).order_by(Step.step_number)
        )
        return result.scalars().all()

    async def create(self, data):
        step = Step(**data.dict())
        self.db.add(step)
        await self.db.commit()
        await self.db.refresh(step)
        return step

    async def update(self, step, data):
        for key, value in data.dict(exclude_unset=True).items():
            setattr(step, key, value)

        await self.db.commit()
        await self.db.refresh(step)
        return step

    async def delete(self, step):
        await self.db.delete(step)
        await self.db.commit()
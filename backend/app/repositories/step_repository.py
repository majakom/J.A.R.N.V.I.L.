from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.element import Element
from models.step import Step
from sqlalchemy.orm import selectinload

class StepRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(self):
        result = await self.db.execute(select(Step).options(selectinload(Step.parts)))
        return result.scalars().all()

    async def get_by_instruction_and_number(self, instruction_id: int, step_number: int):
        result = await self.db.execute(
            select(Step).options(selectinload(Step.parts)).where(Step.instruction_id == instruction_id, Step.step_number == step_number)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, step_id: int):
        result = await self.db.execute(
            select(Step).options(selectinload(Step.parts)).where(Step.id == step_id)
        )
        return result.scalar_one_or_none()

    async def get_by_instruction(self, instruction_id: int):
        result = await self.db.execute(
            select(Step).options(selectinload(Step.parts)).where(Step.instruction_id == instruction_id).order_by(Step.step_number)
        )
        return result.scalars().all()

    async def create(self, data):
        data_dict = data.dict()

        part_ids = data_dict.pop("part_ids")
    
        result = await self.db.execute(
            select(Element).where(Element.id.in_(part_ids))
        )

        parts = result.scalars().all()

        step = Step(**data_dict)
        step.parts = parts

        self.db.add(step)

        await self.db.commit()
        return await self.get_by_id(step.id)

    async def update(self, step, data):
        data_dict = data.dict(exclude_unset=True)

        if "part_ids" in data_dict:
            part_ids = data_dict.pop("part_ids")

            result = await self.db.execute(
                select(Element).where(Element.id.in_(part_ids))
            )

            step.parts = result.scalars().all()

        for key, value in data_dict.items():
            setattr(step, key, value)

        await self.db.commit()
        return await self.get_by_id(step.id)

    async def delete(self, step):
        await self.db.delete(step)
        await self.db.commit()
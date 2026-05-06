from fastapi import Depends
from db.session import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from repositories.element_repository import ElementRepository
from services.element_service import ElementService
from repositories.instruction_repository import InstructionRepository
from services.instruction_service import InstructionService
from repositories.step_repository import StepRepository
from services.step_service import StepService
from typing import AsyncGenerator

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def get_element_service(
    db: AsyncSession = Depends(get_db)
) -> ElementService:
    repo = ElementRepository(db)
    return ElementService(repo)


def get_instruction_service(
    db: AsyncSession = Depends(get_db)
) -> InstructionService:
    repo = InstructionRepository(db)
    return InstructionService(repo)


def get_step_service(
    db: AsyncSession = Depends(get_db)
) -> StepService:
    repo = StepRepository(db)
    return StepService(repo)
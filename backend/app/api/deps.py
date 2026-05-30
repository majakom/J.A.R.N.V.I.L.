from fastapi import Depends
from repositories.yolo_class_repository import YoloClassRepository
from db.session import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from repositories.element_repository import ElementRepository
from services.element_service import ElementService
from services.instruction_generation_service import InstructionGeneratorService
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
    yolo_class_repo = YoloClassRepository(db)
    return ElementService(repo, yolo_class_repo)


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

def get_instruction_generator_service(
    db: AsyncSession = Depends(get_db),
) -> InstructionGeneratorService:
    repo = ElementRepository(db)
    return InstructionGeneratorService(repo)
from fastapi import APIRouter
from api.endpoints import element, instruction, step

api_router = APIRouter()
api_router.include_router(element.router, prefix="/elements", tags=["elements"])
api_router.include_router(instruction.router, prefix="/instructions", tags=["instructions"])
api_router.include_router(step.router, prefix="/steps", tags=["steps"])
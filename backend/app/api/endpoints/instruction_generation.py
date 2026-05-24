from fastapi import APIRouter, Depends
from fastapi_utils.cbv import cbv
from api.deps import get_instruction_generator_service
from schemas.ai_instruction import RetrieveRequest
from services.instruction_generation_service import InstructionGeneratorService

router = APIRouter()

@cbv(router) 
class InstructionGenerationEndpoints:
    instruction_generation_service: InstructionGeneratorService = Depends(get_instruction_generator_service)
    
    @router.post("/simple-parts-retrieve")
    async def simple_parts_retrieve(self, request: RetrieveRequest):
        elements = await self.instruction_generation_service.retrieve_parts_simple(request.concept)

        return [
            {
                "id": element.id,
                "name": element.name,
                "comment": element.comment
            }
            for element in elements
        ]
    
    @router.post("/semantic-parts-retrieve")
    async def semantic_parts_retrieve(self, request: RetrieveRequest):
        elements = await self.instruction_generation_service.retrieve_parts_semantic(request.concept)

        return [
            {
                "id": element.id,
                "name": element.name,
                "comment": element.comment
            }
            for element in elements
        ]
    
    @router.post("/generate")
    async def generate_instruction(self, request: RetrieveRequest):
        return await self.instruction_generation_service.generate_instruction(request.concept)
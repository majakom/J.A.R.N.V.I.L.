from fastapi import APIRouter, Depends, HTTPException
from fastapi_utils.cbv import cbv
from schemas.instruction import InstructionCreate, InstructionUpdate, InstructionRead
from schemas.step import StepCreate, StepUpdate, StepRead
from services.instruction_service import InstructionService
from services.step_service import StepService
from api.deps import get_instruction_service, get_step_service

router = APIRouter()

@cbv(router)
class InstructionEndpoints:
    instruction_service: InstructionService = Depends(get_instruction_service)
    step_service: StepService = Depends(get_step_service)

    @router.get("/", response_model=list[InstructionRead])
    async def get_instructions(self):
        return await self.instruction_service.get_all()

    @router.get("/{instruction_id}", response_model=InstructionRead)
    async def get_instruction(self, instruction_id: int):
        try:
            return await self.instruction_service.get(instruction_id)
        except ValueError:
            raise HTTPException(404, "Not found")

    @router.post("/", response_model=InstructionRead)
    async def create_instruction(self, data: InstructionCreate):
        return await self.instruction_service.create(data)

    @router.put("/{instruction_id}", response_model=InstructionRead)
    async def update_instruction(self, instruction_id: int, data: InstructionUpdate):
        try:
            return await self.instruction_service.update(instruction_id, data)
        except ValueError:
            raise HTTPException(404, "Not found")

    @router.delete("/{instruction_id}")
    async def delete_instruction(self, instruction_id: int):
        try:
            await self.instruction_service.delete(instruction_id)
            return {"message": "deleted"}
        except ValueError:
            raise HTTPException(404, "Not found")

    # Nested steps endpoints
    @router.get("/{instruction_id}/steps", response_model=list[StepRead])
    async def get_steps_for_instruction(self, instruction_id: int):
        return await self.step_service.get_by_instruction(instruction_id)
    
    @router.get("/{instruction_id}/steps/{step_number}", response_model=StepRead)
    async def get_step_for_instruction(self, instruction_id: int, step_number: int):
        step = await self.step_service.get_by_instruction_and_number(instruction_id, step_number)
        if not step:
            raise HTTPException(404, "Step not found")
        return step

    @router.post("/{instruction_id}/steps", response_model=StepRead)
    async def create_step_for_instruction(self, instruction_id: int, data: StepCreate):
        if data.instruction_id != instruction_id:
            raise HTTPException(400, "Instruction ID mismatch")
        return await self.step_service.create(data)

    @router.put("/{instruction_id}/steps/{step_number}", response_model=StepRead)
    async def update_step_for_instruction(self, instruction_id: int, step_number: int, data: StepUpdate):
        step = await self.step_service.get_by_instruction_and_number(instruction_id, step_number)
        if not step:
            raise HTTPException(404, "Step not found")
        return await self.step_service.update(step.id, data)

    @router.delete("/{instruction_id}/steps/{step_number}")
    async def delete_step_for_instruction(self, instruction_id: int, step_number: int):
        step = await self.step_service.get_by_instruction_and_number(instruction_id, step_number)
        if not step:
            raise HTTPException(404, "Step not found")
        await self.step_service.delete(step.id)
        return {"message": "deleted"}
    
    @router.get("/{instruction_id}/current_step", response_model=StepRead)
    async def get_current_step_for_instruction(self, instruction_id: int):
        instruction = await self.instruction_service.get(instruction_id)
        if not instruction:
            raise HTTPException(404, "Instruction not found")
        if not instruction.current_step_id:
            raise HTTPException(404, "Current step not set")
        step = await self.step_service.get(instruction.current_step_id)
        if not step:
            raise HTTPException(404, "Current step not found")
        return step
    
    @router.get("/{instruction_id}/next_step", response_model=StepRead)
    async def get_next_step_for_instruction(self, instruction_id: int):
        instruction = await self.instruction_service.get(instruction_id)
        if not instruction:
            raise HTTPException(404, "Instruction not found")
        if not instruction.current_step_id:
            raise HTTPException(404, "Current step not set")
        current_step = await self.step_service.get(instruction.current_step_id)
        if not current_step:
            raise HTTPException(404, "Current step not found")
        next_step = await self.step_service.get_by_instruction_and_number(instruction_id, current_step.step_number + 1)
        if not next_step:
            raise HTTPException(404, "Next step not found")
        await self.instruction_service.set_current_step(instruction_id, next_step.id)
        return next_step
    
    @router.get("/{instruction_id}/previous_step", response_model=StepRead)
    async def get_previous_step_for_instruction(self, instruction_id: int):
        instruction = await self.instruction_service.get(instruction_id)
        if not instruction:
            raise HTTPException(404, "Instruction not found")
        if not instruction.current_step_id:
            raise HTTPException(404, "Current step not set")
        current_step = await self.step_service.get(instruction.current_step_id)
        if not current_step:
            raise HTTPException(404, "Current step not found")
        previous_step = await self.step_service.get_by_instruction_and_number(instruction_id, current_step.step_number - 1)
        if not previous_step:
            raise HTTPException(404, "Previous step not found")
        await self.instruction_service.set_current_step(instruction_id, previous_step.id)
        return previous_step
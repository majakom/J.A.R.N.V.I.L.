import { api } from "../client/client";
import type { Instruction, GeneratedInstruction, FirstDraftInstruction } from "../../types/instruction";
import type { FirstDraftStep } from "../../types/step";

// =========================
// INSTRUCTIONS
// =========================

export const getInstructions = () =>
    api<Instruction[]>("/api/instructions/");

export const getInstruction = (id: number) =>
    api<Instruction>(`/api/instructions/${id}`);

export const createInstruction = (data: Partial<Instruction>) =>
    api<Instruction>("/api/instructions/", {
        method: "POST",
        body: JSON.stringify(data),
    });

export const updateInstruction = (
    id: number,
    data: Partial<Instruction>
) =>
    api<Instruction>(`/api/instructions/${id}`, {
        method: "PUT",
        body: JSON.stringify(data),
    });

export const deleteInstruction = (id: number) =>
    api<void>(`/api/instructions/${id}`, {
        method: "DELETE",
    });

export const createInstructionWithGenerated = async (data: GeneratedInstruction) => {
  return api<FirstDraftInstruction>(
    "/api/instruction-generation/generate",
    {
      method: "POST",
      body: JSON.stringify(data),
    }
  );
};

export const createWithGeneratedInstruction = async (
  instruction: FirstDraftInstruction
) => {
  return api<FirstDraftInstruction>(
    "/api/instructions/create_with_generated_instruction",
    {
      method: "POST",
      body: JSON.stringify(instruction),
    }
  );
};

// =========================
// STEPS (nested under instructions)
// =========================

export const getNextStep = (id: number) =>
    api<any>(`/api/instructions/${id}/next_step`);

export const getPreviousStep = (id: number) =>
    api<any>(`/api/instructions/${id}/previous_step`);

export const getInstructionStep = (
    instructionId: number,
    stepNumber: number
) =>
    api<any>(
        `/api/instructions/${instructionId}/steps/${stepNumber}`
    );

export const updateStep = (
    instructionId: number,
    stepNumber: number,
    data: any
) =>
    api<any>(
        `/api/instructions/${instructionId}/steps/${stepNumber}`,
        {
            method: "PUT",
            body: JSON.stringify(data),
        }
    );

export const deleteStep = (
    instructionId: number,
    stepNumber: number
) =>
    api<void>(
        `/api/instructions/${instructionId}/steps/${stepNumber}`,
        {
            method: "DELETE",
        }
    );


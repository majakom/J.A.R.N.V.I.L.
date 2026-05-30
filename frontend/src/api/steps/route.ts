import { api } from "../client/client";
import type { Step } from "../../types/step";

export const getSteps = () =>
    api<Step[]>("/api/steps/");

export const getStep = (id: number) =>
    api<Step>(`/api/steps/${id}`);

export const createStep = (data: Partial<Step>) =>
    api<Step>("/api/steps/", {
        method: "POST",
        body: JSON.stringify(data),
    });

export const updateStep = (
    id: number,
    data: Partial<Step>
) =>
    api<Step>(`/api/steps/${id}`, {
        method: "PUT",
        body: JSON.stringify(data),
    });

export const deleteStep = (id: number) =>
    api<void>(`/api/steps/${id}`, {
        method: "DELETE",
    });

export const getInstructionSteps = (
    instructionId: number
) =>
    api<Step[]>(
        `/api/instructions/${instructionId}/steps`
    );

export const getInstructionStep = (
    instructionId: number,
    stepNumber: number
) =>
    api<Step>(
        `/api/instructions/${instructionId}/steps/${stepNumber}`
    );

export const createInstructionStep = (
    instructionId: number,
    data: Partial<Step>
) =>
    api<Step>(
        `/api/instructions/${instructionId}/steps`,
        {
            method: "POST",
            body: JSON.stringify(data),
        }
    );

export const updateInstructionStep = (
    instructionId: number,
    stepNumber: number,
    data: Partial<Step>
) =>
    api<Step>(
        `/api/instructions/${instructionId}/steps/${stepNumber}`,
        {
            method: "PUT",
            body: JSON.stringify(data),
        }
    );

export const deleteInstructionStep = (
    instructionId: number,
    stepNumber: number
) =>
    api<void>(
        `/api/instructions/${instructionId}/steps/${stepNumber}`,
        {
            method: "DELETE",
        }
    );

export const getNextStep = (
    instructionId: number
) =>
    api<Step>(
        `/api/instructions/${instructionId}/next_step`
    );

export const getPreviousStep = (
    instructionId: number
) =>
    api<Step>(
        `/api/instructions/${instructionId}/previous_step`
    );

export const getCurrentStep = (id: number) =>
  api<Step>(`/api/instructions/${id}/current_step/`);

export const showStep = (stepId: number) =>
    api<void>(`/api/steps/${stepId}/show`, {
        method: "POST",
    });
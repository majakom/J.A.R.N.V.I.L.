import type { FirstDraftStep } from "./step";


export interface Instruction {
    id: number;
    name: string;
    current_step_id: number | null;
}

export interface FirstDraftInstruction {
    name: string;
    steps: FirstDraftStep[];
}

export type GeneratedInstruction = {
  concept: string;
};
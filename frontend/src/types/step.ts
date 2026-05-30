export interface Step {
  id: number;
  instruction_id: number;
  step_number: number;
  description: string;
  part_ids: number[];
}

export interface FirstDraftStep {
  step_number: number;
  description: string;
  part_ids: number[];
}
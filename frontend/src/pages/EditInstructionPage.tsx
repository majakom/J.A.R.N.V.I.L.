import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import {
  getInstruction,
  updateInstruction,
  updateStep,
  deleteStep,
} from "../api/instructions/route";

import { getInstructionSteps,
         createInstructionStep,
 } from "../api/steps/route";
import { getElements, updateElement } from "../api/elements/route";

import type { Instruction } from "../types/instruction";
import type { Step } from "../types/step";
import type { Element } from "../types/element";

export default function EditInstruction() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [instruction, setInstruction] = useState<Instruction | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [elements, setElements] = useState<Element[]>([]);
  const [elementSearch, setElementSearch] = useState("");
  const [newStepDraft, setNewStepDraft] = useState({
    description: "",
    part_ids: [] as number[],
    });

  const [loading, setLoading] = useState(true);
  const [nameError, setNameError] = useState("");
  const [stepsError, setStepsError] = useState("");

  useEffect(() => {
    if (!id) return;

    const load = async () => {
      try {
        const [i, s, e] = await Promise.all([
          getInstruction(Number(id)),
          getInstructionSteps(Number(id)),
          getElements(),
        ]);

        setInstruction(i);
        setSteps([...s].sort((a, b) => a.step_number - b.step_number));
        setElements(e);
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [id]);

  const filteredElements = elements.filter((e) =>
    e.name.toLowerCase().includes(elementSearch.toLowerCase())
  );

  const searchableElements = elements.filter((el) =>
    el.name?.toLowerCase().includes(elementSearch.toLowerCase())
  );

  const updateStepField = (stepId: number, field: keyof Step, value: any) => {
    setSteps((prev) =>
      prev.map((s) => (s.id === stepId ? { ...s, [field]: value } : s))
    );
  };

  

  const handleCreateStep = async () => {
    if (!instruction) return;

    try {
      const nextStepNumber =
        steps.length > 0
          ? Math.max(...steps.map((s) => s.step_number)) + 1
          : 1;

      const created = await createInstructionStep(instruction.id, {
        instruction_id: instruction.id,
        step_number: nextStepNumber,
        description: newStepDraft.description,
        part_ids: newStepDraft.part_ids,
      });

      setSteps((prev) =>
        [...prev, created].sort((a, b) => a.step_number - b.step_number)
      );

      setStepsError(""); 

      setNewStepDraft({
        description: "",
        part_ids: [],
      });
    } catch (err) {
      console.error(err);
    }
  };

  const toggleElementForNewStep = (elementId: number) => {
    setNewStepDraft((prev) => {
        const exists = prev.part_ids.includes(elementId);

        return {
        ...prev,
        part_ids: exists
            ? prev.part_ids.filter((id) => id !== elementId)
            : [...prev.part_ids, elementId],
        };
    });
    };

  const handleDeleteStep = async (stepNumber: number) => {
    if (!instruction) return;

    if (steps.length <= 1) {
      alert("Instruction must have at least one step.");
      return;
    }

    try {
      await deleteStep(instruction.id, stepNumber);

      setSteps((prev) => {
        const updated = prev
          .filter((s) => s.step_number !== stepNumber)
          .sort((a, b) => a.step_number - b.step_number);

        return updated;
      });
    } catch (err) {
      console.error(err);
    }
  };


  const toggleElementForStep = (stepId: number, elementId: number) => {
    setSteps((prev) =>
      prev.map((step) => {
        if (step.id !== stepId) return step;

        const exists = step.part_ids.includes(elementId);

        return {
          ...step,
          part_ids: exists
            ? step.part_ids.filter((id) => id !== elementId)
            : [...step.part_ids, elementId],
        };
      })
    );
  };

  const handleSave = async () => {
    if (!instruction) return;

    let hasError = false;

    if (!instruction.name.trim()) {
      setNameError("Instruction name is required.");
      hasError = true;
    }

    if (steps.length < 1) {
      setStepsError("Instruction must contain at least one step.");
      hasError = true;
    }

    if (hasError) return;

    setNameError("");
    setStepsError("");

    await updateInstruction(instruction.id, {
      name: instruction.name,
    });

    for (const step of steps) {
      await updateStep(instruction.id, step.step_number, {
        description: step.description,
        part_ids: step.part_ids,
        step_number: step.step_number,
      });
    }

    for (const el of elements) {
      await updateElement(el.id, {
        name: el.name,
        amount: el.amount,
        url: el.url,
        comment: el.comment,
      });
    }

    alert("Saved");
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#01010a] text-violet-300">
        Loading...
      </div>
    );
  }

  return (
    <div className="min-h-screen flex bg-[#01010a] relative overflow-hidden font-mono text-violet-200">

      {/* BACKGROUND */}
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage:
            "radial-gradient(rgba(139,92,246,0.6) 1px, transparent 1px), radial-gradient(rgba(99,102,241,0.25) 1px, transparent 1px)",
          backgroundSize: "22px 22px, 44px 44px",
          backgroundPosition: "0 0, 11px 11px",
        }}
      />

      <div className="absolute w-[42rem] h-[42rem] bg-violet-950/70 blur-[140px] top-10 left-10" />
      <div className="absolute w-[48rem] h-[48rem] bg-violet-900/40 blur-[160px] bottom-10 right-10" />
      <div className="absolute w-[55rem] h-[55rem] bg-black/80 blur-[180px] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />

      <div className="flex-1 flex items-center justify-center p-10">

        {/* FRAME */}
        <div className="
          relative w-full max-w-4xl p-[1px] rounded-xl
          bg-gradient-to-r from-violet-950 via-black to-violet-900
          shadow-[0_0_80px_rgba(139,92,246,0.20)]
        ">

          <div className="bg-[#01010a]/95 border border-violet-500/30 rounded-xl p-8">

            {/* HEADER */}
            <div className="flex justify-between mb-8">
              <button
                onClick={() => navigate(-1)}
                className="px-4 py-2 border border-violet-500/40 rounded hover:bg-violet-500/10"
              >
                ← BACK
              </button>

              <button
                onClick={handleSave}
                className="px-6 py-2 bg-violet-700 rounded hover:bg-violet-600"
              >
                SAVE
              </button>
            </div>

            {/* TITLE */}
            <div className="mb-8">
              <p className="text-xs text-violet-500 mb-2 tracking-[0.25em]">
                INSTRUCTION NAME
              </p>

              <input
                value={instruction?.name ?? ""}
                onChange={(e) => {
                  setInstruction((prev) =>
                    prev ? { ...prev, name: e.target.value } : null
                  );
                  if (e.target.value.trim()) setNameError("");
                }}
                className={`w-full p-3 bg-black border rounded ${
                  nameError
                    ? "border-red-500 focus:border-red-400"
                    : "border-violet-500/30"
                }`}
              />

              {nameError && (
                <p className="text-red-400 text-sm mt-2">
                  {nameError}
                </p>
              )}
            </div>

            {/* STEPS ERROR */}
            {stepsError && (
              <div className="mb-6 p-3 border border-red-500/40 bg-red-500/10 text-red-400 rounded">
                {stepsError}
              </div>
            )}

            {/* STEPS */}
            <div className="space-y-8">
              {steps.map((step) => (
                <div
                  key={step.id}
                  className="border border-violet-500/20 rounded-xl p-6 bg-black/40"
                >

                  {/* STEP HEADER */}
                  <div className="flex justify-between mb-4">
                    <input
                      type="number"
                      value={step.step_number}
                      onChange={(e) =>
                        updateStepField(
                          step.id,
                          "step_number",
                          Number(e.target.value)
                        )
                      }
                      className="w-24 p-2 bg-black border border-violet-500/30 rounded"
                    />

                    <button
                      onClick={() => handleDeleteStep(step.step_number)}
                      disabled={steps.length <= 1}
                      className={`px-3 py-1 border rounded ${
                        steps.length <= 1
                          ? "border-gray-600 text-gray-600 cursor-not-allowed"
                          : "border-red-500/40 text-red-400"
                      }`}
                    >
                      DELETE
                    </button>
                  </div>

                  {/* DESCRIPTION */}
                  <textarea
                    value={step.description}
                    onChange={(e) =>
                      updateStepField(
                        step.id,
                        "description",
                        e.target.value
                      )
                    }
                    className="w-full min-h-[120px] p-3 bg-black border border-violet-500/30 rounded mb-6"
                  />

                  {/* ASSIGNED */}
                  <p className="text-xs text-violet-500 mb-2">
                    ASSIGNED ELEMENTS
                  </p>

                  <div className="flex flex-wrap gap-2 mb-6">
                    {elements
                      .filter((e) =>
                        step.part_ids.includes(e.id)
                      )
                      .map((e) => (
                        <div
                          key={e.id}
                          className="px-3 py-1 bg-violet-500/10 border border-violet-500/30 rounded flex gap-2"
                        >
                          {e.name}
                          <button
                            onClick={() =>
                              toggleElementForStep(step.id, e.id)
                            }
                            className="text-red-400"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                  </div>

                  {/* SEARCH HEADER DIVIDER */}
                  <div className="my-6 flex items-center gap-3">
                    <div className="h-px flex-1 bg-violet-500/30" />
                    <span className="text-xs text-violet-500 tracking-[0.25em]">
                      SEARCH ELEMENTS
                    </span>
                    <div className="h-px flex-1 bg-violet-500/30" />
                  </div>

                  {/* SEARCH */}
                  <input
                    value={elementSearch}
                    onChange={(e) =>
                      setElementSearch(e.target.value)
                    }
                    placeholder="Search elements..."
                    className="w-full p-2 bg-black border border-violet-500/30 rounded mb-4"
                  />

                  {/* RESULTS */}
                  <div className="max-h-64 overflow-y-auto space-y-2">
                    {filteredElements.map((el) => {
                      const assigned = step.part_ids.includes(el.id);

                      return (
                        <div
                          key={el.id}
                          className="flex justify-between border border-violet-500/20 p-3 rounded"
                        >
                          <div>
                            <p>{el.name}</p>
                            <p className="text-xs text-violet-500">
                              {el.amount}
                            </p>
                          </div>

                          <button
                            onClick={() =>
                              toggleElementForStep(step.id, el.id)
                            }
                            className={`px-3 py-1 rounded border ${
                              assigned
                                ? "border-green-500 text-green-400"
                                : "border-violet-500 text-violet-300"
                            }`}
                          >
                            {assigned ? "Assigned" : "Assign"}
                          </button>
                        </div>
                      );
                    })}
                  </div>

                </div>
              ))}
            </div>
            {/* CREATE NEW STEP */}
            <div className="border border-violet-500/40 rounded-xl p-6 bg-black/30 mt-10">
                <p className="text-xs text-violet-500 mb-4 tracking-[0.25em]">
                    ADD NEW STEP
                </p>

                {/* DESCRIPTION */}
                <textarea
                    value={newStepDraft.description}
                    onChange={(e) =>
                    setNewStepDraft((prev) => ({
                        ...prev,
                        description: e.target.value,
                    }))
                    }
                    placeholder="Step description..."
                    className="w-full min-h-[120px] p-3 bg-black border border-violet-500/30 rounded mb-6"
                />

                {/* SELECTED PARTS */}
                <p className="text-xs text-violet-500 mb-2">
                    SELECTED ELEMENTS
                </p>

                <div className="flex flex-wrap gap-2 mb-6">
                    {elements
                    .filter((e) => newStepDraft.part_ids.includes(e.id))
                    .map((e) => (
                        <div
                        key={e.id}
                        className="px-3 py-1 bg-violet-500/10 border border-violet-500/30 rounded flex gap-2"
                        >
                        {e.name}
                        <button
                            onClick={() => toggleElementForNewStep(e.id)}
                            className="text-red-400"
                        >
                            ×
                        </button>
                        </div>
                    ))}
                </div>
                {/* SEARCH ELEMENTS */}
                <div className="mb-4">
                  <input
                    type="text"
                    value={elementSearch}
                    onChange={(e) => setElementSearch(e.target.value)}
                    placeholder="Search elements..."
                    className="
                      w-full
                      bg-black/40
                      border border-violet-500/20
                      rounded-lg
                      px-3 py-2
                      text-violet-100
                      placeholder:text-violet-500/50
                      focus:outline-none
                      focus:border-violet-400/60
                      transition
                    "
                  />
                </div>

                {/* ELEMENT LIST */}
                <div className="max-h-64 overflow-y-auto space-y-2 mb-6">
                  {searchableElements.length === 0 ? (
                    <div className="text-center py-4 text-violet-500">
                      No matching elements found
                    </div>
                  ) : (
                    searchableElements.map((el) => {
                      const assigned = newStepDraft.part_ids.includes(el.id);

                      return (
                        <div
                          key={el.id}
                          className="flex justify-between border border-violet-500/20 p-3 rounded"
                        >
                          <div>
                            <p>{el.name}</p>
                            <p className="text-xs text-violet-500">{el.amount}</p>
                          </div>

                          <button
                            onClick={() => toggleElementForNewStep(el.id)}
                            className={`px-3 py-1 rounded border ${
                              assigned
                                ? "border-green-500 text-green-400"
                                : "border-violet-500 text-violet-300"
                            }`}
                          >
                            {assigned ? "Selected" : "Select"}
                          </button>
                        </div>
                      );
                    })
                  )}
                </div>

                {/* CREATE BUTTON */}
                <button
                    onClick={handleCreateStep}
                    className="px-6 py-2 bg-green-600 rounded hover:bg-green-500"
                >
                    CREATE STEP
                </button>
                </div>

          </div>
        </div>
      </div>
    </div>
  );
}
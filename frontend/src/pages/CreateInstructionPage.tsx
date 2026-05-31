import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { createInstruction } from "../api/instructions/route";
import { createInstructionStep } from "../api/steps/route";
import { getElements } from "../api/elements/route";

import type { Element } from "../types/element";

export default function CreateInstruction() {
  const navigate = useNavigate();

  const [instructionName, setInstructionName] = useState("");
  const [elements, setElements] = useState<Element[]>([]);
  const [elementSearch, setElementSearch] = useState("");

  const [steps, setSteps] = useState<
    {
      tempId: string;
      step_number: number;
      description: string;
      part_ids: number[];
    }[]
  >([]);

  const [newStepDraft, setNewStepDraft] = useState({
    description: "",
    part_ids: [] as number[],
  });

  const [loading, setLoading] = useState(true);

  const [errors, setErrors] = useState<{
    name?: string;
    steps?: string;
  }>({});

  useEffect(() => {
    const load = async () => {
      try {
        const e = await getElements();
        setElements(e);
      } finally {
        setLoading(false);
      }
    };

    load();
  }, []);

  

  const filteredElements = elements.filter((e) =>
    e.name.toLowerCase().includes(elementSearch.toLowerCase())
  );

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

  const handleAddStep = () => {
    if (!newStepDraft.description.trim()) return;

    const nextStepNumber =
      steps.length > 0
        ? Math.max(...steps.map((s) => s.step_number)) + 1
        : 1;

    setSteps((prev) => [
      ...prev,
      {
        tempId: crypto.randomUUID(),
        step_number: nextStepNumber,
        description: newStepDraft.description,
        part_ids: newStepDraft.part_ids,
      },
    ]);

    setErrors((prev) => ({ ...prev, steps: undefined }));

    setNewStepDraft({
      description: "",
      part_ids: [],
    });
  };

  const handleRemoveStep = (tempId: string) => {
    setSteps((prev) =>
      prev
        .filter((s) => s.tempId !== tempId)
        .map((s, idx) => ({
          ...s,
          step_number: idx + 1,
        }))
    );
  };

  const toggleElementForStep = (tempId: string, elementId: number) => {
    setSteps((prev) =>
      prev.map((step) => {
        if (step.tempId !== tempId) return step;

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

  const handleCreate = async () => {
    const newErrors: { name?: string; steps?: string } = {};

    if (!instructionName.trim()) {
      newErrors.name = "Instruction name is required";
    }

    if (steps.length === 0) {
      newErrors.steps = "At least one step is required";
    }

    setErrors(newErrors);

    if (Object.keys(newErrors).length > 0) return;

    try {
      const instruction = await createInstruction({
        name: instructionName,
      });

      for (const step of steps) {
        await createInstructionStep(instruction.id, {
          instruction_id: instruction.id,
          step_number: step.step_number,
          description: step.description,
          part_ids: step.part_ids,
        });
      }

      navigate(-1);
    } catch (err) {
      console.error(err);
    }
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
                onClick={handleCreate}
                className="px-6 py-2 bg-violet-600 rounded hover:bg-violet-500"
              >
                CREATE
              </button>
            </div>

            {/* NAME */}
            <div className="mb-8">
              <p className="text-xs text-violet-500 mb-2 tracking-[0.25em]">
                INSTRUCTION NAME
              </p>
              {errors.name && (
                <p className="text-red-400 text-xs mt-2">
                  {errors.name}
                </p>
              )}

              <input
                value={instructionName}
                onChange={(e) => {
                  setInstructionName(e.target.value);

                  if (errors.name) {
                    setErrors((prev) => ({ ...prev, name: undefined }));
                  }
                }}
                className="w-full p-3 bg-black border border-violet-500/30 rounded"
              />
            </div>

            {/* STEPS */}
            <div className="space-y-6 mb-10">
              {steps.map((step) => (
                <div
                  key={step.tempId}
                  className="border border-violet-500/20 rounded-xl p-6 bg-black/40"
                >
                  <div className="flex justify-between mb-4">
                    <span className="text-sm text-violet-400">
                      STEP {step.step_number}
                    </span>

                    <button
                      onClick={() => handleRemoveStep(step.tempId)}
                      className="px-3 py-1 border border-red-500/40 text-red-400 rounded"
                    >
                      DELETE
                    </button>
                  </div>

                  <p className="mb-4">{step.description}</p>

                  {/* assigned */}
                  <div className="flex flex-wrap gap-2">
                    {elements
                      .filter((e) => step.part_ids.includes(e.id))
                      .map((e) => (
                        <div
                          key={e.id}
                          className="px-3 py-1 bg-violet-500/10 border border-violet-500/30 rounded flex gap-2"
                        >
                          {e.name}
                          <button
                            onClick={() =>
                              toggleElementForStep(step.tempId, e.id)
                            }
                            className="text-red-400"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                  </div>
                </div>
              ))}
            </div>
            {errors.steps && (
              <p className="text-red-400 text-xs mb-4">
                {errors.steps}
              </p>
            )}

            {/* ADD STEP */}
            <div className="border border-violet-500/30 rounded-xl p-6 bg-black/30 mb-10">
              <p className="text-xs text-violet-500 mb-4 tracking-[0.25em]">
                ADD STEP
              </p>

              <textarea
                value={newStepDraft.description}
                onChange={(e) =>
                  setNewStepDraft((p) => ({
                    ...p,
                    description: e.target.value,
                  }))
                }
                className="w-full min-h-[120px] p-3 bg-black border border-violet-500/30 rounded mb-4"
              />

              <button
                onClick={handleAddStep}
                className="px-4 py-2 bg-violet-700 rounded hover:bg-violet-600"
              >
                ADD STEP
              </button>
            </div>

            {/* ELEMENTS FOR NEW STEP */}
            <div className="border border-violet-500/20 rounded-xl p-6 bg-black/30">
              <p className="text-xs text-violet-500 mb-2">
                SELECT ELEMENTS
              </p>

              <input
                value={elementSearch}
                onChange={(e) => setElementSearch(e.target.value)}
                placeholder="Search..."
                className="w-full p-2 bg-black border border-violet-500/30 rounded mb-4"
              />

              <div className="max-h-64 overflow-y-auto space-y-2">
                {filteredElements.map((el) => {
                  const assigned =
                    newStepDraft.part_ids.includes(el.id);

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
                          toggleElementForNewStep(el.id)
                        }
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
                })}
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { getInstructions } from "../api/instructions/route";
import {
  getInstructionSteps,
  getNextStep,
  getPreviousStep,
  getCurrentStep,
  updateCurrentStep,
} from "../api/steps/route";

import { setElementShow } from "../api/elements/route";
import { getElements } from "../api/elements/route";
import { useYoloStream } from "../api/yolo_stream/route";
import type { Instruction } from "../types/instruction";
import type { Step } from "../types/step";
import type { Element } from "../types/element";

export default function Dashboard() {
  const navigate = useNavigate();
  const { id } = useParams();

  // =========================
  // INSTRUCTIONS
  // =========================
  const [instructions, setInstructions] = useState<Instruction[]>([]);
  const [loadingInstructions, setLoadingInstructions] = useState(true);
  const streamUrl = useYoloStream();
  const [selectedInstruction, setSelectedInstruction] =
    useState<Instruction | null>(null);

  // =========================
  // STEPS
  // =========================
  const [steps, setSteps] = useState<Step[]>([]);
  const [activeStep, setActiveStep] = useState<Step | null>(null);
  const [loadingSteps, setLoadingSteps] = useState(false);

  // =========================
  // ELEMENTS
  // =========================
  const [elements, setElements] = useState<Element[]>([]);
  const [selectedElement, setSelectedElement] =
    useState<Element | null>(null);

  // =========================
  // LOAD DATA
  // =========================
  useEffect(() => {
    const fetchInstructions = async () => {
      setLoadingInstructions(true);
      try {
        const data = await getInstructions();
        setInstructions(data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoadingInstructions(false);
      }
    };

    fetchInstructions();
  }, []);

  useEffect(() => {
    const loadElements = async () => {
      try {
        const data = await getElements();
        setElements(data);
      } catch (err) {
        console.error(err);
      }
    };

    loadElements();
  }, []);

  useEffect(() => {
    if (!id || instructions.length === 0) return;

    const instruction = instructions.find(
      (ins) => ins.id === Number(id)
    );

    if (instruction) {
      setSelectedInstruction(instruction);
      fetchSteps(instruction.id);
      loadCurrentStep(instruction.id);
    }
  }, [id, instructions]);

  // =========================
  // STEPS
  // =========================
  const fetchSteps = async (instructionId: number) => {
    try {
      setLoadingSteps(true);

      const data = await getInstructionSteps(instructionId);

      const sorted = [...data].sort(
        (a, b) => a.step_number - b.step_number
      );

      setSteps(sorted);

      if (sorted.length > 0) setActiveStep(sorted[0]);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingSteps(false);
    }
  };

  const loadCurrentStep = async (instructionId: number) => {
    try {
      const step = await getCurrentStep(instructionId);
      setActiveStep(step);
    } catch (err) {
      console.error(err);
    }
  };

  // reset selection on step change
  useEffect(() => {
    setSelectedElement(null);
  }, [activeStep]);

  // =========================
  // NAVIGATION
  // =========================
  const goNext = async () => {
    if (!selectedInstruction) return;
    const step = await getNextStep(selectedInstruction.id);
    if (step) setActiveStep(step);
  };

  const goPrev = async () => {
    if (!selectedInstruction) return;
    const step = await getPreviousStep(selectedInstruction.id);
    if (step) setActiveStep(step);
  };

  // =========================
  // ELEMENT ACTIONS
  // =========================
  const handleShowElement = async (id: number) => {
    await setElementShow(id, true);
  };

  const handleStopShowElement = async (id: number) => {
    await setElementShow(id, false);
  };

  // =========================
  // DERIVE STEP ELEMENTS
  // =========================
  const stepElements =
    activeStep
      ? elements.filter((el) =>
          activeStep.part_ids.includes(el.id)
        )
      : [];

  return (
    <div className="min-h-screen flex bg-[#01010a] relative overflow-hidden font-mono text-violet-200">

      {/* BACKGROUND (RESTORED) */}
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

      {/* ========================= */}
      {/* SIDEBAR (RESTORED STYLE) */}
      {/* ========================= */}
      <div className="w-72 p-6 border-r border-violet-500/20 bg-black/30 backdrop-blur-md z-10">

        <button
          onClick={() => navigate("/instructions_database")}
          className="w-full py-2 mb-6 rounded-md bg-black/40 border border-violet-500/30 text-violet-300 tracking-[0.25em] text-xs hover:border-violet-400/60 transition"
        >
          ← INSTRUCTIONS
        </button>

        <h3 className="text-violet-300 tracking-[0.35em] text-xs mb-4">
          STEPS
        </h3>

        {loadingSteps ? (
          <p className="text-xs text-violet-500">Loading steps...</p>
        ) : (
          <div className="space-y-3">
            {steps.map((step) => (
              <div
                key={step.id}
                onClick={() => {setActiveStep(step);
                                updateCurrentStep(selectedInstruction!.id, step.id);}
                }
                className={`cursor-pointer border p-3 rounded-md transition ${
                  activeStep?.id === step.id
                    ? "border-violet-400 bg-violet-500/10"
                    : "border-violet-500/20 bg-black/40 hover:border-violet-400/50"
                }`}
              >
                <p className="text-xs text-violet-500">
                  STEP {step.step_number}
                </p>
                <p className="text-sm text-violet-200">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ========================= */}
      {/* MAIN PANEL (RESTORED UI) */}
      {/* ========================= */}
      <div className="flex-1 flex items-center justify-center pt-12 pb-20">

        <div className="
          relative w-full max-w-2xl p-[1px] rounded-xl
          bg-gradient-to-r from-violet-950 via-black to-violet-900
          shadow-[0_0_80px_rgba(139,92,246,0.20)]
        ">

          <div className="bg-[#01010a]/95 border border-violet-500/30 rounded-xl p-8">

            {/* HEADER */}
            <div className="text-center mb-8 space-y-3">
              <p className="text-violet-100 text-lg tracking-wide">
                {selectedInstruction?.name ?? "No instruction selected"}
              </p>

              {activeStep && (
                <p className="text-xs text-violet-600 tracking-[0.25em]">
                  STEP {activeStep.step_number}
                </p>
              )}
            </div>

            {/* ACTIVE STEP BOX */}
            <div className="border border-violet-600/30 bg-black/60 rounded-md p-10 mb-6">

              {activeStep ? (
                <>
                  <p className="text-violet-300 text-sm leading-relaxed whitespace-pre-wrap mb-6">
                    {activeStep.description}
                  </p>

                  {/* ELEMENTS */}
                  <div className="space-y-2 mb-6">
                    <p className="text-xs text-violet-500 tracking-[0.25em]">
                      ELEMENTS
                    </p>

                    {stepElements.length === 0 ? (
                      <p className="text-xs text-violet-600 italic">
                        No elements assigned to this step
                      </p>
                    ) : (
                      stepElements.map((el) => (
                        <div
                          key={el.id}
                          onClick={() => setSelectedElement(el)}
                          className={`cursor-pointer p-2 border rounded transition ${
                            selectedElement?.id === el.id
                              ? "border-violet-400 bg-violet-500/10"
                              : "border-violet-500/20 hover:border-violet-400/40"
                          }`}
                        >
                          <p className="text-sm text-violet-200">{el.name}</p>
                          <p className="text-xs text-violet-500">
                            amount: {el.amount}
                          </p>
                        </div>
                      ))
                    )}
                  </div>

                  {/* ACTIONS */}
                  <div className="flex gap-4">
                    <button
                      disabled={!selectedElement}
                      onClick={() =>
                        selectedElement &&
                        handleShowElement(selectedElement.id)
                      }
                      className="text-violet-300 px-3 py-1 border border-violet-400 rounded hover:bg-violet-400/10 transition disabled:opacity-40"
                    >
                      SHOW ON TABLE
                    </button>

                    <button
                      disabled={!selectedElement}
                      onClick={() =>
                        selectedElement &&
                        handleStopShowElement(selectedElement.id)
                      }
                      className="text-cyan-300 px-3 py-1 border border-cyan-400 rounded hover:bg-cyan-400/10 transition disabled:opacity-40"
                    >
                      STOP SHOWING
                    </button>
                  </div>
                </>
              ) : (
                <p className="text-violet-500">Select a step</p>
              )}
            </div>

            {/* NAVIGATION */}
            <div className="flex gap-4 w-full">

              <button
                onClick={goPrev}
                className="w-full py-3 rounded-md bg-black/70 border border-violet-500/50 text-violet-200 tracking-[0.35em]"
              >
                &lt; PREVIOUS STEP
              </button>

              <button
                onClick={goNext}
                className="w-full py-3 rounded-md bg-black/70 border border-violet-500/50 text-violet-200 tracking-[0.35em]"
              >
                NEXT STEP &gt;
              </button>

            </div>

          </div>

          {/* === YOLO LIVE STREAM BOX === */}
          <div className="md:col-span-2 border border-violet-500/40 bg-black/70 p-4 rounded-xl backdrop-blur-md ">

            <p className="text-violet-400 text-[10px] mb-3 uppercase tracking-[0.35em]">
              Live Camera Feed (YOLO)
            </p>

            <div className="relative w-full border border-violet-900/50 rounded overflow-hidden bg-black flex items-center justify-center">
              
              <img
                src={streamUrl}
                alt="YOLO Stream"
                className="max-w-full h-auto object-contain"
              />

            </div>

          </div>
          
        </div>
        
      </div>
    </div>
  );
}
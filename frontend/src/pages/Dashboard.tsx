import { useNavigate } from "react-router-dom";
import { useState, useEffect, useMemo } from "react";

import {
  getInstructions,
  createInstructionWithGenerated,
  createWithGeneratedInstruction,
  getNextStep,
  getPreviousStep,
} from "../api/instructions/route";

import { 
  getElements 
} from "../api/elements/route";

import {
  getCurrentStep,
} from "../api/steps/route";

import type { Step } from "../types/step";
import type { Element } from "../types/element";
import type { Instruction, FirstDraftInstruction } from "../types/instruction";
import type { FirstDraftStep } from "../types/step";
import { getCamera } from "../api/camera/route";
import type { Camera } from "../types/camera";


export default function Dashboard() {
  const navigate = useNavigate();

  const [showDraftModal, setShowDraftModal] = useState(false);
  const [generatedDraft, setGeneratedDraft] = useState<FirstDraftStep[]>([]);
  const [instructionName, setInstructionName] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState(false);

  // =========================
  // NAVIGATION
  // =========================
  const logout = () => {
    localStorage.removeItem("token");
    navigate("/");
  };

  const select_database = () => {
    navigate("/select_database");
  };

  // =========================
  // CAMERA STATE
  // =========================
  const [cameraStatus, setCameraStatus] = useState<boolean | null>(null);

  // =========================
  // STATE
  // =========================
  const [instructions, setInstructions] = useState<Instruction[]>([]);
  const [loadingInstructions, setLoadingInstructions] = useState(true);
  const [selectedInstruction, setSelectedInstruction] =
    useState<Instruction | null>(null);
  const [stepData, setStepData] = useState<Step | null>(null);
  const [elements, setElements] = useState<Element[]>([]);
  const [loadingStep, setLoadingStep] = useState(false);

  // =========================
  // INSTRUCTION TEXT
  // =========================
  const [instructionText, setInstructionText] = useState("");

  // =========================
  // LOAD INSTRUCTIONS
  // =========================
  const fetchInstructions = async () => {
    try {
      const data = await getInstructions();
      setInstructions(data.slice(0, 3));
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    const load = async () => {
      setLoadingInstructions(true);
      await fetchInstructions();
      setLoadingInstructions(false);
    };

    load();
  }, []);

  // =========================
  // LOAD STEP (FIXED)
  // =========================
  const loadStep = async (instructionId: number) => {
    try {
      setLoadingStep(true);

      const data = await getCurrentStep(instructionId);

      setStepData(data);
    } catch (err) {
      console.error(err);
      setStepData(null);
    } finally {
      setLoadingStep(false);
    }
  };
  
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

  const elementById = useMemo(
    () => new Map(elements.map(el => [el.id, el])),
    [elements]
  );
  
  // =========================
  // NAVIGATION STEPS (FIXED)
  // =========================
  const goNext = async () => {
    if (!selectedInstruction) return;

    try {
      const step = await getNextStep(selectedInstruction.id);
      if (step) setStepData(step);
    } catch (err) {
      console.error(err);
    }
  };

  const goPrev = async () => {
    if (!selectedInstruction) return;

    try {
      const step = await getPreviousStep(selectedInstruction.id);
      if (step) setStepData(step);
    } catch (err) {
      console.error(err);
    }
  };

  // =========================
  // CAMERA STATUS
  // =========================
  const fetchCamera = async () => {
    try {
      const data: Camera[] = await getCamera();
      setCameraStatus(data[0]?.status ?? false);
    } catch (err) {
      console.error(err);
      setCameraStatus(false);
    }
  };

  useEffect(() => {
    fetchCamera();
  }, []);


  const acceptDraft = async () => {
    try {
      await createWithGeneratedInstruction({
        name: instructionName,
        steps: generatedDraft,
      });

      setShowDraftModal(false);
      setGeneratedDraft([]);
      setInstructionName("");
      setInstructionText("");

      await fetchInstructions();
    } catch (err) {
      console.error(err);
    }
  };

  // =========================
  // UI
  // =========================
  return (
    <div className="min-h-screen bg-[#01010a] relative overflow-hidden font-mono text-violet-200 p-8">

      {/* GRID BACKGROUND */}
      <div
        className="absolute inset-0 opacity-20 pointer-events-none"
        style={{
          backgroundImage:
            "radial-gradient(rgba(139,92,246,0.6) 1px, transparent 1px), radial-gradient(rgba(99,102,241,0.25) 1px, transparent 1px)",
          backgroundSize: "22px 22px, 44px 44px",
          backgroundPosition: "0 0, 11px 11px",
        }}
      />

      {/* FOG */}
      <div className="absolute w-[42rem] h-[42rem] bg-violet-950/30 blur-[140px] top-10 left-10"></div>
      <div className="absolute w-[48rem] h-[48rem] bg-violet-900/20 blur-[160px] bottom-10 right-10"></div>

      {/* GRID */}
      <div className="relative z-10 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-7xl mx-auto">

        {/* HEADER */}
        <div className="md:col-span-2 border border-violet-500/30 bg-black/60 p-8 rounded-xl backdrop-blur-md">
          <h1 className="text-violet-100 text-3xl tracking-[0.4em] mb-2">
            J.A.R.N.V.I.L.
          </h1>

          <p
            className={`text-xs tracking-[0.35em] uppercase ${
              cameraStatus ? "text-green-400" : "text-red-400"
            }`}
          >
            Camera Status: {cameraStatus ? "Connected" : "Disconnected"}
          </p>
        </div>

        {/* SIDEBAR */}
        <div className="flex flex-col gap-4">
          <div className="border border-red-500/30 bg-black/60 p-6 rounded-xl">
            <button
              onClick={logout}
              className="text-red-400 text-xs tracking-[0.52em] hover:text-red-200 transition text-left"
            >
              [ DISCONNECT_SESSION ]
            </button>
          </div>

          <div className="border border-red-500/30 bg-black/60 p-6 rounded-xl">
            <button
              onClick={select_database}
              className="text-red-400 text-xs tracking-[0.7em] hover:text-red-200 transition text-left"
            >
              [ DATABASE_ACCESS ]
            </button>
          </div>
        </div>

        {/* LEFT PANEL */}
        <div className="flex flex-col gap-4">

          {/* INPUT */}
          <div className="flex flex-col gap-4 p-4 border border-violet-900/30 rounded-xl bg-black/40 backdrop-blur-md hover:border-violet-500/30 transition">

            <div className="flex items-center justify-between">
              <span className="text-xs tracking-[0.4em] text-violet-500 uppercase">
                Project Creator
              </span>
              <span className="w-2 h-2 rounded-full bg-violet-500/60 shadow-[0_0_10px_rgba(139,92,246,0.5)]" />
            </div>

            <div className="h-32 border border-violet-900/20 rounded-lg bg-black/30 p-3 focus-within:border-violet-500/40 transition">
              <textarea
                value={instructionText}
                onChange={(e) => setInstructionText(e.target.value)}
                placeholder="What project are we starting?"
                className="w-full h-full bg-transparent text-violet-100 text-sm tracking-[0.15em] outline-none resize-none placeholder:text-violet-900/60"
              />
            </div>

            <button
              onClick={async () => {
                if (!instructionText.trim() || isGenerating) return;

                setIsGenerating(true);

                try {
                  const draft: FirstDraftInstruction =
                    await createInstructionWithGenerated({
                      concept: instructionText,
                    });

                  setInstructionName(draft.name);
                  setGeneratedDraft(draft.steps);
                  setShowDraftModal(true);
                } catch (err) {
                  console.error(err);
                } finally {
                  setIsGenerating(false);
                }
              }}
              disabled={isGenerating}
              className={`px-4 py-1.5 text-xs tracking-[0.5em] rounded-md border transition flex items-center gap-2
                ${
                  isGenerating
                    ? "bg-black/60 border-violet-500/40 text-violet-400 cursor-not-allowed"
                    : "bg-black/70 border-violet-900/40 text-violet-300 hover:bg-black hover:border-violet-500/40"
                }`}
            >
              {isGenerating ? (
                <>
                  <span className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                  GENERATING...
                </>
              ) : (
                "GENERATE INSTRUCTION"
              )}
            </button>
          </div>

          {/* INSTRUCTIONS */}
          <div className="flex flex-col gap-4">
            {loadingInstructions ? (
              <p className="text-violet-400 text-sm">Loading...</p>
            ) : (
              instructions.map((ins, index) => (
                <div
                  key={ins.id}
                  onClick={async () => {
                    setSelectedInstruction(ins);
                    await loadStep(ins.id);
                  }}
                  className="border border-violet-500/20 bg-black/40 p-4 rounded-lg cursor-pointer hover:border-violet-400/40 transition"
                >
                  <p className="text-xs text-violet-500 tracking-widest">
                    {String(index + 1).padStart(2, "0")}
                  </p>
                  <p className="text-lg text-violet-100">{ins.name}</p>
                  
                </div>
              ))
            )}
          </div>
        </div>

        {/* MAIN PANEL */}
        <div className="md:col-span-2 border border-violet-600/50 bg-black/80 p-6 rounded-xl shadow-[0_0_30px_rgba(139,92,246,0.12)] flex flex-col gap-6">

          <div className="flex items-start justify-between pb-4 border-b border-violet-500/20">
            <div>
              <span className="text-xs tracking-[0.4em] text-violet-400 uppercase">
                Project
              </span>

              <p className="text-3xl text-violet-100 tracking-[0.2em] mt-1">
                {selectedInstruction?.name || "—"}
              </p>
            </div>

            <div>
              {selectedInstruction && (
                <button
                  onClick={() =>
                    navigate(`/show_instruction/${selectedInstruction.id}`)
                  }
                  className="px-3 py-1.5 text-xs tracking-[0.4em] text-violet-200 border border-violet-400/70 bg-black/60 rounded-md hover:bg-black hover:border-violet-300 transition"
                >
                  GO TO FOCUS MODE
                </button>
              )}
            </div>
          </div>

          <div className="h-64 border border-violet-500/40 rounded bg-violet-950/10 p-5 flex flex-col justify-between">
            {loadingStep ? (
              <p>Loading step...</p>
            ) : !selectedInstruction ? (
              <p>Select instruction</p>
            ) : (
              <>
                <p className="text-xs text-violet-400 tracking-[0.3em] uppercase">
                  STEP {stepData?.step_number ?? "-"}
                </p>

                <p className="text-sm text-violet-100 mt-3 leading-relaxed tracking-[0.1em]">
                  {stepData?.description || "No step data available"}
                </p>

                {stepData?.part_ids.length != 0 && <div className="mt-3">
                  <p className="text-sm text-violet-500 leading-relaxed tracking-[0.1em]">
                    Parts:
                  </p>

                  {stepData?.part_ids?.map((id) => {
                     const el = elementById.get(id);

                      if (!el) return null;

                      return (
                        <div key={id} className="text-sm">
                          <p className="text-violet-200">{el.name}</p>
                        </div>
                      );
                    })}
                </div>}
              </>
            )}
          </div>

          <div className="flex gap-4">
            <button
              onClick={goPrev}
              className="w-full py-3 rounded-md bg-black/70 border border-violet-500/50 text-violet-200 text-xs tracking-[0.4em] hover:border-violet-300 hover:bg-black transition"
            >
              &lt; PREV
            </button>

            <button
              onClick={goNext}
              className="w-full py-3 rounded-md bg-black/70 border border-violet-500/50 text-violet-200 text-xs tracking-[0.4em] hover:border-violet-300 hover:bg-black transition"
            >
              NEXT &gt;
            </button>
          </div>

        </div>
      </div>
      {showDraftModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">

          <div className="w-[900px] max-h-[85vh] overflow-y-auto bg-[#050510] border border-violet-500/40 rounded-xl p-6 shadow-[0_0_40px_rgba(139,92,246,0.3)]">

            <div className="flex justify-between items-center mb-6">

              <h2 className="text-xl text-violet-100 tracking-[0.3em] uppercase">
                Generated Instruction
              </h2>

              <button
                onClick={() => setShowDraftModal(false)}
                className="text-red-400 hover:text-red-300"
              >
                ✕
              </button>
            </div>
            <div className="mb-6 border-b border-violet-500/20 pb-4">
              <p className="text-xs tracking-[0.4em] text-violet-400 uppercase">
                Instruction Name
              </p>

              <p className="text-2xl text-violet-100 mt-2 tracking-[0.2em]">
                {instructionName || "Unnamed Instruction"}
              </p>
            </div>

            <div className="space-y-4">

              {generatedDraft.map((step) => (
                  <div
                    key={step.step_number}
                    className="border border-violet-500/20 rounded-lg p-4 bg-black/40"
                  >
                    <p className="text-violet-400 text-xs tracking-[0.3em] uppercase">
                      Step {step.step_number}
                    </p>

                    <p className="text-violet-100 mt-2">
                      {step.description}
                    </p>

                    {step?.part_ids?.length > 0 && (
                      <div className="mt-3">
                        <p className="text-violet-500 text-xs uppercase mb-2">
                          Parts
                        </p>
                      </div>
                    )}
                      <div className="flex flex-wrap gap-2">
                        
                        {step.part_ids.map((partId) => (
                          <span
                            key={partId}
                            className="px-2 py-1 rounded bg-violet-900/40 border border-violet-500/30 text-xs text-violet-200"
                          >
                            #{partId}
                          </span>
                        ))}
                      </div>
                    </div>
                ))}
            </div>

            <div className="flex gap-4 mt-8">

              <button
                onClick={() => setShowDraftModal(false)}
                className="flex-1 py-3 border border-red-500/40 rounded-md text-red-300 hover:bg-red-950/20"
              >
                Reject
              </button>

              <button
                onClick={acceptDraft}
                className="flex-1 py-3 border border-green-500/40 rounded-md text-green-300 hover:bg-green-950/20"
              >
                Accept Instruction
              </button>

            </div>

          </div>

        </div>
      )}
    </div>
  );
}
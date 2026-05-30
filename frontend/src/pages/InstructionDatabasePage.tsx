import { useEffect, useState } from "react";
import {
  getInstructions,
  updateInstruction,
  deleteInstruction,
} from "../api/instructions/route";

import { useNavigate } from "react-router-dom";

import type { Instruction } from "../types/instruction";

export default function InstructionsPanel() {
  // =========================
  // INSTRUCTIONS STATE
  // =========================
  const [instructions, setInstructions] = useState<Instruction[]>([]);
  const [loadingInstructions, setLoadingInstructions] = useState(true);

  const [selectedInstruction, setSelectedInstruction] =
    useState<Instruction | null>(null);

  const [editingInstructionId, setEditingInstructionId] =
    useState<number | null>(null);

  const [instructionEditValue, setInstructionEditValue] =
    useState("");

  const navigate = useNavigate();

  // =========================
  // LOAD INSTRUCTIONS
  // =========================
  const fetchInstructions = async () => {
    try {
      setLoadingInstructions(true);

      const data = await getInstructions();

      setInstructions(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingInstructions(false);
    }
  };

  useEffect(() => {
    fetchInstructions();
  }, []);

  // =========================
  // DELETE
  // =========================
  const handleInstructionDelete = async (id: number) => {
    try {
      await deleteInstruction(id);

      await fetchInstructions();

      if (selectedInstruction?.id === id) {
        setSelectedInstruction(null);
        setEditingInstructionId(null);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // =========================
  // UPDATE
  // =========================
  const handleInstructionUpdate = async (id: number) => {
    try {
      await updateInstruction(id, {
        name: instructionEditValue,
      });

      setEditingInstructionId(null);
      setInstructionEditValue("");

      await fetchInstructions();

      if (selectedInstruction?.id === id) {
        setSelectedInstruction({
          ...selectedInstruction,
          name: instructionEditValue,
        });
      }
    } catch (err) {
      console.error(err);
    }
  };

  // =========================
  // UI
  // =========================
  return (
    <div className="min-h-screen bg-[#01010a] relative overflow-hidden font-mono text-violet-200 p-10">

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
      <div className="absolute w-[42rem] h-[42rem] bg-violet-950/30 blur-[140px] top-10 left-10" />
      <div className="absolute w-[48rem] h-[48rem] bg-violet-900/20 blur-[160px] bottom-10 right-10" />

      <div className="relative z-10 max-w-7xl mx-auto">

        {/* TITLE */}
        <div className="mb-16">
          <div className="flex flex-col items-start">
            <button
              onClick={() => navigate("/select_database")}
              className="inline-flex items-center py-1 px-3 mb-13 rounded-md bg-black/40 border border-violet-500/30 text-violet-300 tracking-[0.2em] text-xs hover:border-violet-400/60 transition w-fit"
            >
              ← CONTROL PANEL
            </button>
          </div>
          <div className="border-l-2 border-violet-500 pl-4 mb-10">
            <h2 className="text-2xl tracking-[0.3em] text-violet-100">
              INSTRUCTIONS
            </h2>

            <p className="text-xs text-violet-500 tracking-[0.3em] mt-1">
              CONTROL PANEL
            </p>
          </div>

          {loadingInstructions ? (
            <p className="text-violet-400">
              Loading instructions...
            </p>
          ) : (
            <div className="flex gap-6">

              {/* LEFT GRID */}
              <div className="flex-1 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">

                {instructions.map((ins, index) => (
                  <div
                    key={ins.id}
                    onClick={() => setSelectedInstruction(ins)}
                    className={`
                      cursor-pointer
                      border
                      border-violet-500/20
                      bg-black/40
                      p-5
                      rounded-xl
                      backdrop-blur-md
                      hover:border-violet-400/40
                      transition
                      duration-300
                      hover:shadow-[0_0_30px_rgba(139,92,246,0.12)]

                      ${
                        selectedInstruction?.id === ins.id
                          ? "border-violet-400"
                          : ""
                      }
                    `}
                  >

                    <p className="text-xs text-violet-500 tracking-widest">
                      {String(index + 1).padStart(2, "0")}
                    </p>

                    <p className="text-lg text-violet-100 mt-3 tracking-wide">
                      {ins.name}
                    </p>

                  </div>
                ))}
              </div>

              {/* SEPARATOR */}
              {selectedInstruction && (
                <div className="w-px bg-violet-500/20" />
              )}

              {/* RIGHT DETAILS PANEL */}
              {selectedInstruction && (
                <div className="w-[380px] border border-violet-500/20 bg-black/40 p-5 rounded-xl backdrop-blur-md">

                  {/* HEADER */}
                  <div className="flex justify-between items-center mb-6">

                    <h3 className="text-violet-200 tracking-[0.3em] text-sm">
                      DETAILS
                    </h3>

                    <button
                      onClick={() => setSelectedInstruction(null)}
                      className="
                        text-red-400
                        w-8
                        h-8
                        flex
                        items-center
                        justify-center
                        hover:bg-red-500/10
                        rounded
                        transition
                      "
                    >
                      ×
                    </button>

                  </div>

                  {/* NAME / EDIT */}
                  {editingInstructionId === selectedInstruction.id ? (
                    <input
                      value={instructionEditValue}
                      onChange={(e) =>
                        setInstructionEditValue(e.target.value)
                      }
                      className="
                        w-full
                        bg-black/60
                        border
                        border-violet-500/30
                        rounded
                        px-3
                        py-2
                        text-sm
                        outline-none
                      "
                    />
                  ) : (
                    <p className="text-lg text-violet-100">
                      {selectedInstruction.name}
                    </p>
                  )}

                  {/* START INSTRUCTION */}
                  <div className="flex gap-4 mt-4">

                    <button
                      onClick={() =>
                        navigate(
                          `/show_instruction/${selectedInstruction.id}`
                        )
                      }
                      className="
                        text-violet-300
                        px-3
                        py-1
                        border
                        border-violet-400
                        rounded
                        hover:bg-violet-400/10
                        transition
                      "
                    >
                      START INSTRUCTION...
                    </button>

                  </div>

                  {/* SEPARATOR */}
                  <div className="mt-4 border-t border-white/10 pt-4" />

                  {/* ACTIONS */}
                  <div className="flex gap-6 mt-6">

                    {editingInstructionId === selectedInstruction.id ? (
                      <button
                        onClick={() =>
                          handleInstructionUpdate(
                            selectedInstruction.id
                          )
                        }
                        className="
                          text-violet-300
                          px-3
                          py-1
                          border
                          border-violet-400
                          rounded
                          hover:bg-violet-400/10
                          transition
                        "
                      >
                        SAVE
                      </button>
                    ) : (
                      <div className="flex items-center justify-center gap-4 mt-2">

                        <button
                          onClick={() => {
                            setEditingInstructionId(
                              selectedInstruction.id
                            );

                            setInstructionEditValue(
                              selectedInstruction.name
                            );
                          }}
                          className="
                            text-violet-300
                            px-3
                            py-1
                            border
                            border-violet-400
                            rounded
                            hover:bg-violet-400/10
                            transition
                          "
                        >
                          EDIT
                        </button>

                        <button
                          onClick={() =>
                            handleInstructionDelete(
                              selectedInstruction.id
                            )
                          }
                          className="
                            text-red-300
                            px-3
                            py-1
                            border
                            border-red-400
                            rounded
                            hover:bg-red-400/10
                            transition
                          "
                        >
                          DELETE
                        </button>

                      </div>
                    )}

                  </div>

                </div>
              )}

            </div>
          )}
        </div>
      </div>
    </div>
  );
}
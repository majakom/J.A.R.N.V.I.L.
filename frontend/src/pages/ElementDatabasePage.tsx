import { useEffect, useState } from "react";

import {
  getElements,
  createElement,
  updateElement,
  deleteElement,
  setElementShow,
} from "../api/elements/route";
import { useNavigate } from "react-router-dom";
import type { Element } from "../types/element";

export default function InstructionsPanel() {
  const navigate = useNavigate();

  const [elements, setElements] = useState<Element[]>([]);
  const [loadingElements, setLoadingElements] = useState(true);
  const [selectedElement, setSelectedElement] = useState<Element | null>(null);

  // CREATE
  const [newElementName, setNewElementName] = useState("");
  const [newAmount, setNewAmount] = useState<number>(1);
  const [newLink, setNewLink] = useState("");
  const [newComment, setNewComment] = useState("");
  const [isCreateOpen, setIsCreateOpen] = useState(false);

  // EDIT FULL
  const [editingElementId, setEditingElementId] = useState<number | null>(null);
  const [editValues, setEditValues] = useState({
    name: "",
    amount: 1,
    url: "",
    comment: "",
  });

  // LOAD
  const fetchElements = async () => {
    try {
      setLoadingElements(true);
      const data = await getElements();
      setElements(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingElements(false);
    }
  };

  useEffect(() => {
    fetchElements();
  }, []);

  // CREATE
  const handleElementCreate = async () => {
    if (!newElementName.trim()) return;

    await createElement({
      name: newElementName,
      amount: newAmount,
      url: newLink,
      comment: newComment,
    });

    setNewElementName("");
    setNewAmount(1);
    setNewLink("");
    setNewComment("");

    fetchElements();
  };

  // DELETE (NOW ONLY FROM DETAILS)
  const handleElementDelete = async (id: number) => {
    await deleteElement(id);
    fetchElements();

    if (selectedElement?.id === id) {
      setSelectedElement(null);
    }
  };

  // SHOW / STOP
  const handleShowElement = async (id: number) => {
    await setElementShow(id, true);
  };

  const handleStopShowElement = async (id: number) => {
    await setElementShow(id, false);
  };

  // EDIT
  const startEdit = (el: Element) => {
    setEditingElementId(el.id);

    setEditValues({
      name: el.name || "",
      amount: el.amount || 1,
      url: el.url || "",
      comment: el.comment || "",
    });
  };

  const saveEdit = async (id: number) => {
    await updateElement(id, editValues);

    setEditingElementId(null);
    fetchElements();

    if (selectedElement?.id === id) {
      setSelectedElement({ ...selectedElement, ...editValues });
    }
  };

  return (
    <div className="min-h-screen bg-[#01010a] relative overflow-hidden font-mono text-violet-200 p-10">

      {/* BACKGROUND */}
      <div
        className="absolute inset-0 opacity-20 pointer-events-none"
        style={{
          backgroundImage:
            "radial-gradient(rgba(139,92,246,0.6) 1px, transparent 1px), radial-gradient(rgba(99,102,241,0.25) 1px, transparent 1px)",
          backgroundSize: "22px 22px, 44px 44px",
          backgroundPosition: "0 0, 11px 11px",
        }}
      />

      <div className="absolute w-[42rem] h-[42rem] bg-violet-950/30 blur-[140px] top-10 left-10" />
      <div className="absolute w-[48rem] h-[48rem] bg-violet-900/20 blur-[160px] bottom-10 right-10" />

      <div className="relative z-10 max-w-7xl mx-auto">

        {/* TITLE */}
        
        <div className="mb-16">
          <div className="flex flex-col items-start">
            <button
              onClick={() => navigate("/select_database")}
              className="inline-flex items-center py-1 px-3 mb-13 rounded-md bg-black/40 border border-cyan-500/30 text-cyan-300 tracking-[0.2em] text-xs hover:border-cyan-400/60 transition w-fit"
            >
              ← CONTROL PANEL
            </button>
          </div>
          <div className="mb-8 border-l-2 border-cyan-500 pl-4">
            <h2 className="text-2xl tracking-[0.3em] text-cyan-100">
              ELECTRIC PARTS
            </h2>
            <p className="text-xs text-cyan-500 tracking-[0.3em] mt-1">
              CONTROL PANEL
            </p>
          </div>
        </div>

        {/* CREATE */}
        <div className="mb-10">
          <div
            onClick={() => setIsCreateOpen((p) => !p)}
            className="cursor-pointer border border-cyan-500/20 bg-black/40 rounded-xl p-5 backdrop-blur-md hover:border-cyan-400/40 transition"
          >
            <div className="flex justify-between items-center">
              <div>
                <p className="text-cyan-300 tracking-[0.3em] text-sm">
                  ADD ELECTRONIC PART
                </p>
              </div>
              <div className="text-xl">{isCreateOpen ? "×" : "+"}</div>
            </div>
          </div>

          {isCreateOpen && (
            <div className="mt-4 border border-cyan-500/20 bg-black/30 rounded-xl p-6">
              <div className="grid grid-cols-2 gap-4">
                <input
                  value={newElementName}
                  onChange={(e) => setNewElementName(e.target.value)}
                  placeholder="Name"
                  className="bg-black/60 p-2 rounded"
                />
                <input
                  type="number"
                  value={newAmount}
                  onChange={(e) => setNewAmount(Number(e.target.value))}
                  placeholder="Amount"
                  className="bg-black/60 p-2 rounded"
                />
                <input
                  value={newLink}
                  onChange={(e) => setNewLink(e.target.value)}
                  placeholder="Link"
                  className="bg-black/60 p-2 rounded"
                />
                <input
                  value={newComment}
                  onChange={(e) => setNewComment(e.target.value)}
                  placeholder="Comment"
                  className="bg-black/60 p-2 rounded"
                />
              </div>

              <button
                onClick={handleElementCreate}
                className="mt-4 px-4 py-2 border border-cyan-400"
              >
                ADD
              </button>
            </div>
          )}
        </div>

        {/* MAIN LAYOUT */}
        <div className="flex gap-6">

          {/* LEFT CARDS (NO DELETE HERE ANYMORE) */}
          <div className="flex-1 grid grid-cols-3 gap-4">
            {loadingElements ? (
              <p>Loading...</p>
            ) : (
              elements.map((el, i) => (
                <div
                  key={el.id}
                  onClick={() => setSelectedElement(el)}
                  className={`cursor-pointer border rounded-xl p-5 bg-black/40 transition
                    ${selectedElement?.id === el.id
                      ? "border-cyan-400 shadow-[0_0_25px_rgba(34,211,238,0.25)]"
                      : "border-cyan-500/20 hover:border-cyan-400/40"
                    }`}
                >
                  <p className="text-xs text-cyan-500">{i + 1}</p>
                  <p className="text-cyan-100">{el.name}</p>

                </div>
              ))
            )}
          </div>

          {/* SEPARATOR */}
          {selectedElement && (
            <div className="w-px bg-cyan-500/20" />
          )}

          {/* RIGHT DETAILS CARD */}
          {selectedElement && (
            <div className="w-[400px]">

              <div className="border border-cyan-500/20 bg-black/40 rounded-xl p-6 backdrop-blur-md">

                <div className="flex justify-between mb-6">
                  <h3 className="text-cyan-200 tracking-[0.3em] text-sm">
                    DETAILS
                  </h3>
                  {/* DELETE (ONLY HERE NOW) */}
                  

                  <button
                    onClick={() => setSelectedElement(null)}
                    className="text-red-400 text-xs px-3 py-1 rounded hover:bg-red-500/10 transition"
                  >
                    x
                  </button>
                  
                </div>

                <div className="mt-4 border-t border-white/10 pt-4"/>
                

                {/* VIEW / EDIT */}
                {editingElementId === selectedElement.id ? (
                  <div className="space-y-3">

                    <input
                      placeholder="Enter name..."
                      value={editValues.name}
                      onChange={(e) =>
                        setEditValues({ ...editValues, name: e.target.value })
                      }
                      className="w-full bg-black/60 p-2 rounded"
                    />

                    <input
                      placeholder="Enter amount..."
                      type="number"
                      value={editValues.amount}
                      onChange={(e) =>
                        setEditValues({
                          ...editValues,
                          amount: Number(e.target.value),
                        })
                      }
                      className="w-full bg-black/60 p-2 rounded"
                    />

                    <input
                      placeholder="Add link..."
                      value={editValues.url}
                      onChange={(e) =>
                        setEditValues({ ...editValues, url: e.target.value })
                      }
                      className="w-full bg-black/60 p-2 rounded"
                    />

                    <textarea
                      placeholder="Add description..."
                      value={editValues.comment}
                      onChange={(e) =>
                        setEditValues({ ...editValues, comment: e.target.value })
                      }
                      rows={4}
                      className="
                        w-full
                        bg-black/60
                        p-2
                        rounded
                        resize-y
                        min-h-[100px]
                        overflow-y-auto

                        scrollbar-thin
                        scrollbar-track-transparent
                        scrollbar-thumb-cyan-400/40
                        hover:scrollbar-thumb-cyan-300/60
                      "
                    />
                    <div className="flex items-center justify-center mt-2 w-full">
                      <button
                        onClick={() => saveEdit(selectedElement.id)}
                        className="w-full text-cyan-300 px-3 py-2 border border-cyan-400 rounded hover:bg-cyan-400/10 transition"
                      >
                        SAVE
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <p className="text-cyan-100">{selectedElement.name}</p>
                    <p>{selectedElement.amount}</p>

                    {selectedElement.url && (
                      <a
                        href={selectedElement.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-cyan-300 underline break-all"
                      >
                        {selectedElement.url}
                      </a>
                    )}

                    <p>{selectedElement.comment}</p>

                    
                    {/* SHOW / STOP */}
                    <div className="flex gap-4 mt-4">
                      <button
                        onClick={() => handleShowElement(selectedElement.id)}
                        className="text-violet-300 px-3 py-1 border border-violet-400 rounded hover:bg-violet-400/10 transition"
                      >
                        SHOW ON THE TABLE
                      </button>

                      <button
                        onClick={() => handleStopShowElement(selectedElement.id)}
                        className="text-cyan-300 px-3 py-1 border border-cyan-400 rounded hover:bg-cyan-400/10 transition"
                      >
                        STOP SHOWING
                      </button>
                    </div>

                    <div className="mt-4 border-t border-white/10 pt-4"/>

                    <div className="flex items-center justify-center gap-4 mt-2">
                      <button
                        onClick={() => startEdit(selectedElement)}
                        className="text-cyan-300 px-3 py-1 border border-cyan-400 rounded hover:bg-cyan-400/10 transition"
                      >
                        EDIT
                      </button>
                      <button
                        onClick={() => handleElementDelete(selectedElement.id)}
                        className="text-red-300 px-3 py-1 border border-red-400 rounded hover:bg-red-400/10 transition"
                      >
                        DELETE
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
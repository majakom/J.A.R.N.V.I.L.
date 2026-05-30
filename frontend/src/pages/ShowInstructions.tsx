import { useNavigate } from "react-router-dom";

export default function Dashboard() {
  const navigate = useNavigate();

  const logout = () => {
    localStorage.removeItem("token");
    navigate("/");
  };

  // Placeholders for your navigation logic
  const nextStep = () => console.log("Next...");
  const prevStep = () => console.log("Previous...");

  return (
    <div className="min-h-screen bg-[#01010a] relative overflow-hidden font-mono text-violet-200 p-8">
      
      {/* === VIOLET NEURAL GRID BACKGROUND === */}
      <div
        className="absolute inset-0 opacity-20 pointer-events-none"
        style={{
          backgroundImage:
            "radial-gradient(rgba(139,92,246,0.6) 1px, transparent 1px), radial-gradient(rgba(99,102,241,0.25) 1px, transparent 1px)",
          backgroundSize: "22px 22px, 44px 44px",
          backgroundPosition: "0 0, 11px 11px",
        }}
      />

      {/* === ATMOSPHERIC FOG === */}
      <div className="absolute w-[42rem] h-[42rem] bg-violet-950/30 blur-[140px] top-10 left-10 pointer-events-none"></div>
      <div className="absolute w-[48rem] h-[48rem] bg-violet-900/20 blur-[160px] bottom-10 right-10 pointer-events-none"></div>

      {/* === SCATTERED CONTENT GRID === */}
      <div className="relative z-10 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-7xl mx-auto">
        
        {/* HEADER CARD - Takes up full width on small, 2 cols on large */}
        <div className="md:col-span-2 border border-violet-500/30 bg-black/60 p-8 rounded-xl backdrop-blur-md">
          <h1 className="text-violet-100 text-3xl tracking-[0.4em] mb-2">J.A.R.N.V.I.L.</h1>
          <p className="text-violet-500 text-xs tracking-[0.35em] uppercase">Camera Status: Connected</p>
        </div>

        {/* LOGOUT / SETTINGS CARD */}
        <div className="border border-red-500/30 bg-black/60 p-6 rounded-xl backdrop-blur-md flex flex-col justify-center">
          <button 
            onClick={logout}
            className="text-red-400 text-xs tracking-[0.3em] hover:text-red-200 transition text-left"
          >
            [ DISCONNECT_SESSION ]
          </button>
        </div>
        
        
        <div className="border border-violet-500/20 bg-black/40 p-4 rounded-lg">
          <p className="text-violet-500 text-[10px] mb-2 uppercase tracking-widest">Instruction 01</p>
          <p className="text-sm text-violet-200/80 italic">"The last project"</p>
        </div>

        {/* INSTRUCTION CARD 1 */}
        <div className="border border-violet-500/20 bg-black/40 p-4 rounded-lg">
          <p className="text-violet-500 text-[10px] mb-2 uppercase tracking-widest">Instruction 01</p>
          <p className="text-sm text-violet-200/80 italic">"The last project"</p>
        </div>

        {/* MAIN INTERACTIVE PANEL */}
        <div className="md:col-span-2 border border-violet-600/50 bg-black/80 p-6 rounded-xl shadow-[0_0_30px_rgba(139,92,246,0.1)]">
          <div className="mb-6 h-32 border border-violet-900/50 rounded bg-violet-950/10 p-4">
             <span className="text-violet-400 animate-pulse">_</span>
             <p className="text-sm">Awaiting manual override...</p>
          </div>

          {/* THE BUTTONS IN ONE LINE */}
          <div className="flex gap-4 w-full">
            <button
              onClick={prevStep}
              className="w-full py-3 rounded-md bg-black/70 border border-violet-500/50 text-violet-200 tracking-[0.35em] hover:border-violet-300/70 hover:text-white transition"
            >
              &lt; PREV
            </button>
            <button
              onClick={nextStep}
              className="w-full py-3 rounded-md bg-black/70 border border-violet-500/50 text-violet-200 tracking-[0.35em] hover:border-violet-300/70 hover:text-white transition"
            >
              NEXT &gt;
            </button>
          </div>
        </div>

        <div className="border border-violet-500/20 bg-black/40 p-4 rounded-lg">
          <p className="text-violet-500 text-[10px] mb-2 uppercase tracking-widest">ACCESS</p>
        </div>

        {/* SMALL DATA CARDS */}
        <div className="grid grid-cols-2 gap-4 md:col-span-3">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="border border-violet-500/10 bg-black/20 p-3 rounded text-[10px] uppercase tracking-tighter text-violet-400/50">
                Data_Node_0{i} // Stable
              </div>
            ))}
        </div>

      </div>
    </div>
  );
}
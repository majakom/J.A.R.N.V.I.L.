import { useNavigate } from "react-router-dom";

export default function ControlHub() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#01010a] relative overflow-hidden font-mono text-violet-200 flex items-center justify-center p-8">

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
      <div className="absolute w-[48rem] h-[48rem] bg-cyan-900/20 blur-[160px] bottom-10 right-10" />

      {/* CONTENT */}
      <div className="relative z-10 w-full max-w-6xl">
        <div className="flex flex-col items-start">
            <button
              onClick={() => navigate("/dashboard")}
              className="inline-flex items-center py-1 px-3 mb-13 rounded-md bg-black/40 border border-violet-500/30 text-violet-300 tracking-[0.2em] text-xs hover:border-violet-400/60 transition w-fit"
            >
              ← DASHBOARD
            </button>
          </div>
        {/* HEADER */}
        <div className="border border-violet-500/30 bg-black/60 p-10 rounded-2xl backdrop-blur-md mb-12 text-center shadow-[0_0_35px_rgba(139,92,246,0.08)]">
        
    
        <h1 className="text-red-400 text-4xl tracking-[0.5em] hover:text-red-200">
         DATABASE ACCESS 
        </h1>

        </div>

        {/* PANELS */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

          {/* INSTRUCTIONS PANEL */}
          <div
            onClick={() => navigate("/instructions_database")}
            className="group cursor-pointer border border-violet-500/20 bg-black/50 rounded-2xl p-10 backdrop-blur-md transition duration-300 hover:border-violet-400/50 hover:shadow-[0_0_40px_rgba(139,92,246,0.15)] hover:scale-[1.02]"
          >

            <div className="flex items-center justify-between mb-10">
              <div>

                <h2 className="text-3xl text-violet-100 tracking-[0.25em]">
                  INSTRUCTIONS
                </h2>
              </div>

              <div className="w-4 h-4 rounded-full bg-violet-400 animate-pulse" />
            </div>

            <p className="text-sm text-violet-300 leading-7 tracking-wide">
              Access instruction database, manage uploaded procedures,
              update records and control instruction flow.
            </p>

            <div className="mt-10 flex items-center justify-between">

              <span className="text-xs text-violet-500 tracking-[0.35em]">
                OPEN PANEL
              </span>

              <span className="text-violet-300 text-2xl group-hover:translate-x-2 transition-transform duration-300">
                →
              </span>

            </div>

          </div>

          {/* ELECTRONIC PARTS PANEL */}
          <div
            onClick={() => navigate("/elements_database")}
            className="group cursor-pointer border border-cyan-500/20 bg-black/50 rounded-2xl p-10 backdrop-blur-md transition duration-300 hover:border-cyan-400/50 hover:shadow-[0_0_40px_rgba(34,211,238,0.15)] hover:scale-[1.02]"
          >

            <div className="flex items-center justify-between mb-10">
              <div>

                <h2 className="text-3xl text-cyan-100 tracking-[0.25em]">
                  ELECTRONIC PARTS
                </h2>
              </div>

              <div className="w-4 h-4 rounded-full bg-cyan-400 animate-pulse" />
            </div>

            <p className="text-sm text-cyan-300 leading-7 tracking-wide">
              Manage electronic components, quantities, links,
              comments and hardware inventory records.
            </p>

            <div className="mt-10 flex items-center justify-between">

              <span className="text-xs text-cyan-500 tracking-[0.35em]">
                OPEN PANEL
              </span>

              <span className="text-cyan-300 text-2xl group-hover:translate-x-2 transition-transform duration-300">
                →
              </span>

            </div>

          </div>

        </div>

      </div>
    </div>
  );
}

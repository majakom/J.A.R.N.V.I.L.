import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../services/auth";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const res = await login(email, password);
      localStorage.setItem("token", res.token);
      navigate("/dashboard");
    } catch (err: any) {
      setError("ACCESS DENIED");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#02030a] relative overflow-hidden font-mono text-violet-200">

      {/* deep violet-core atmosphere */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(139,92,246,0.18),transparent_55%)]"></div>

      {/* heavy ambient depth layers */}
      <div className="absolute w-[32rem] h-[32rem] bg-violet-950/40 blur-3xl top-10 left-10"></div>
      <div className="absolute w-[34rem] h-[34rem] bg-indigo-950/30 blur-3xl bottom-10 right-10"></div>
      <div className="absolute w-[42rem] h-[42rem] bg-black/60 blur-3xl top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"></div>

      {/* terminal frame */}
      <div className="
        relative w-full max-w-md p-[1px] rounded-xl
        bg-gradient-to-r from-violet-950 via-indigo-950 to-violet-900
        shadow-[0_0_60px_rgba(139,92,246,0.18)]
      ">

        {/* inner terminal */}
        <div className="bg-[#02030a]/95 border border-violet-500/30 rounded-xl p-6">

          {/* header */}
          <div className="text-center mb-6">
            <h1 className="text-violet-200 text-lg tracking-[0.35em]">
              SYSTEM ACCESS
            </h1>
            <p className="text-violet-500 text-xs mt-2 tracking-[0.35em]">
              AUTHORIZATION REQUIRED
            </p>
          </div>


          {/* form */}
          <form onSubmit={handleLogin} className="space-y-4">

            <input
              type="email"
              placeholder="user@core.node"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="
                w-full p-3 rounded-md
                bg-black/70 text-violet-100
                border border-violet-600/40
                outline-none
                focus:border-violet-400
                focus:ring-0
              "
            />

            <input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="
                w-full p-3 rounded-md
                bg-black/70 text-violet-100
                border border-violet-600/40
                outline-none
                focus:border-violet-400
                focus:ring-0
              "
            />

            {error && (
              <p className="text-violet-400 text-xs tracking-widest">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="
                w-full py-3 rounded-md
                bg-black/70
                border border-violet-500/50
                text-violet-200 tracking-[0.35em]
                hover:border-violet-300/70
                transition
                disabled:opacity-40
              "
            >
              {loading ? "AUTHORIZING..." : "> INITIATE ACCESS"}
            </button>
          </form>

        </div>
      </div>
    </div>
  );
}
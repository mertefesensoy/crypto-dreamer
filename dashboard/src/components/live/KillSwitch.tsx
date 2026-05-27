import { useState } from "react";

export function KillSwitch(): JSX.Element {
  const [armed, setArmed] = useState(false);
  const [killed, setKilled] = useState(false);

  if (killed) {
    return (
      <div className="flex h-full items-center justify-between text-xs">
        <span className="text-red-400">KILLED</span>
        <button
          onClick={() => {
            setKilled(false);
            setArmed(false);
          }}
          className="rounded border border-zinc-700 px-2 py-1 text-zinc-400 hover:text-zinc-100"
        >
          reset
        </button>
      </div>
    );
  }

  return (
    <div className="flex h-full items-center justify-between gap-2 text-xs">
      <label className="flex items-center gap-2 text-zinc-400">
        <input
          type="checkbox"
          checked={armed}
          onChange={(e) => setArmed(e.target.checked)}
          className="accent-red-500"
        />
        arm
      </label>
      <button
        disabled={!armed}
        onClick={() => {
          // No-op for v1: there is no remote-control channel back to the
          // agent yet. The button arms a future POST / Redis publish.
          setKilled(true);
        }}
        className={
          "flex-1 rounded px-2 py-1 font-semibold uppercase tracking-wide transition-colors " +
          (armed
            ? "bg-red-500 text-zinc-950 hover:bg-red-400"
            : "cursor-not-allowed bg-zinc-800 text-zinc-600")
        }
      >
        kill
      </button>
    </div>
  );
}

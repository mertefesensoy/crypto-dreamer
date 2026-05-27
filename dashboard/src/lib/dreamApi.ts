import type { DreamEpisode, DreamRequest, DreamResponse } from "@/types";

const API_ROOT = import.meta.env.VITE_DREAMER_API ?? "http://localhost:8000";

export async function listDreamEpisodes(): Promise<DreamEpisode[]> {
  const r = await fetch(`${API_ROOT}/dream/episodes`);
  if (!r.ok) {
    if (r.status === 503) {
      throw new Error("World model not loaded yet (no checkpoint).");
    }
    throw new Error(`GET /dream/episodes -> ${r.status}`);
  }
  return r.json();
}

export async function postDream(req: DreamRequest): Promise<DreamResponse> {
  const r = await fetch(`${API_ROOT}/dream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    if (r.status === 503) {
      throw new Error("World model not loaded yet (no checkpoint).");
    }
    const txt = await r.text().catch(() => "");
    throw new Error(`POST /dream -> ${r.status}: ${txt.slice(0, 200)}`);
  }
  return r.json();
}

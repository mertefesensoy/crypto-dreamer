import { useEffect } from "react";
import { useDreamerStore } from "@/store/useDreamerStore";
import type { EpisodeSummary, StepInfo } from "@/types";

const DEFAULT_API = "http://127.0.0.1:8000";
const DEFAULT_SEED_COUNT = 2000;

/**
 * On mount, fetch the episode summary list and the last N steps of the
 * latest episode. Seeds the store so the dashboard isn't empty when
 * loaded mid-episode. The WebSocket then takes over for live tail.
 */
export function useHistoryHydration(
  apiBase: string = DEFAULT_API,
  count: number = DEFAULT_SEED_COUNT,
): void {
  useEffect(() => {
    const ctrl = new AbortController();
    void (async () => {
      try {
        const epRes = await fetch(`${apiBase}/episodes`, { signal: ctrl.signal });
        if (!epRes.ok) return;
        const summaries = (await epRes.json()) as EpisodeSummary[];
        // Pull the seed methods at call-time so HMR module reloads don't
        // leave us with stale function references.
        const store = useDreamerStore.getState();
        if (summaries.length > 0) store.seedEpisodeSummaries(summaries);

        const latest =
          summaries.length > 0
            ? summaries[summaries.length - 1]!.episode
            : 0;

        const histRes = await fetch(
          `${apiBase}/history?episode=${latest}&count=${count}`,
          { signal: ctrl.signal },
        );
        if (!histRes.ok) return;
        const steps = (await histRes.json()) as StepInfo[];
        if (steps.length > 0) {
          const store2 = useDreamerStore.getState();
          store2.seedHistory(latest, steps);
          store2.setActiveEpisode(latest);
        }
      } catch (e) {
        const err = e as { name?: string; message?: string };
        // Strict Mode unmount aborts in-flight fetches; surfaces as
        // AbortError or "Failed to fetch" depending on browser.
        if (err?.name === "AbortError" || ctrl.signal.aborted) return;
        console.error("[hydration] failed", err?.name, err?.message);
      }
    })();
    return () => {
      ctrl.abort();
    };
  }, [apiBase, count]);
}

/**
 * Imperative loader for the "Load earlier" button: fetches up to
 * `count` more steps strictly before `beforeStep`, merges into the
 * episode bucket. Returns how many steps it added.
 */
export async function loadEarlier(
  episode: number,
  beforeStep: number,
  count: number = DEFAULT_SEED_COUNT,
  apiBase: string = DEFAULT_API,
): Promise<number> {
  const res = await fetch(
    `${apiBase}/history?episode=${episode}&count=${count}&before_step=${beforeStep}`,
  );
  if (!res.ok) return 0;
  const steps = (await res.json()) as StepInfo[];
  if (steps.length === 0) return 0;
  useDreamerStore.getState().seedHistory(episode, steps);
  return steps.length;
}

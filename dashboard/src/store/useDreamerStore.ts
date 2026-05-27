import { create } from "zustand";
import type {
  ConnectionState,
  EpisodeSummary,
  StepInfo,
} from "@/types";

const RING_SIZE = 50_000;

export type EpisodeBucket = {
  episode: number;
  steps: StepInfo[];
  summary: EpisodeSummary | null;
};

type DreamerState = {
  episodes: Record<number, EpisodeBucket>;
  episodeOrder: number[];
  activeEpisode: number | null;
  connection: ConnectionState;
  pulseTick: number;
  totalSteps: number;
  hydrated: boolean;

  pushStep: (s: StepInfo) => void;
  pushSummary: (s: EpisodeSummary) => void;
  setActiveEpisode: (ep: number) => void;
  setConnection: (c: ConnectionState) => void;
  /**
   * Seed an episode bucket from /history. Idempotent on overlap with
   * already-buffered live frames: we de-dupe by `step` number.
   */
  seedHistory: (episode: number, steps: StepInfo[]) => void;
  /**
   * Replace the cached episode summary list from /episodes.
   */
  seedEpisodeSummaries: (summaries: EpisodeSummary[]) => void;
};

function emptyBucket(episode: number): EpisodeBucket {
  return { episode, steps: [], summary: null };
}

function appendBounded(buf: StepInfo[], s: StepInfo): StepInfo[] {
  if (buf.length < RING_SIZE) return buf.concat(s);
  return buf.slice(buf.length - RING_SIZE + 1).concat(s);
}

export const useDreamerStore = create<DreamerState>((set) => ({
  episodes: {},
  episodeOrder: [],
  activeEpisode: null,
  connection: "connecting",
  pulseTick: 0,
  totalSteps: 0,
  hydrated: false,

  pushStep: (s) =>
    set((state) => {
      const existing = state.episodes[s.episode] ?? emptyBucket(s.episode);
      // Drop duplicates that already came in via /history seed.
      if (
        existing.steps.length > 0 &&
        existing.steps[existing.steps.length - 1]!.step >= s.step
      ) {
        // Out-of-order or duplicate: keep stream monotone.
        const dup = existing.steps.some((x) => x.step === s.step);
        if (dup) return { pulseTick: state.pulseTick + 1 };
      }
      const updatedBucket: EpisodeBucket = {
        episode: s.episode,
        steps: appendBounded(existing.steps, s),
        summary: existing.summary,
      };
      const isNew = state.episodes[s.episode] === undefined;
      return {
        episodes: { ...state.episodes, [s.episode]: updatedBucket },
        episodeOrder: isNew
          ? [...state.episodeOrder, s.episode]
          : state.episodeOrder,
        activeEpisode:
          state.activeEpisode === null ? s.episode : state.activeEpisode,
        pulseTick: state.pulseTick + 1,
        totalSteps: state.totalSteps + 1,
      };
    }),

  pushSummary: (s) =>
    set((state) => {
      const existing = state.episodes[s.episode] ?? emptyBucket(s.episode);
      const isNew = state.episodes[s.episode] === undefined;
      return {
        episodes: {
          ...state.episodes,
          [s.episode]: { ...existing, episode: s.episode, summary: s },
        },
        episodeOrder: isNew
          ? [...state.episodeOrder, s.episode]
          : state.episodeOrder,
      };
    }),

  setActiveEpisode: (ep) => set({ activeEpisode: ep }),
  setConnection: (c) => set({ connection: c }),

  seedHistory: (episode, steps) =>
    set((state) => {
      const existing = state.episodes[episode] ?? emptyBucket(episode);
      // Merge: keep existing in-memory steps, prepend any history steps
      // we don't already have (by step number).
      const have = new Set(existing.steps.map((x) => x.step));
      const fresh = steps.filter((x) => !have.has(x.step));
      const merged = [...fresh, ...existing.steps].sort(
        (a, b) => a.step - b.step,
      );
      const trimmed =
        merged.length <= RING_SIZE
          ? merged
          : merged.slice(merged.length - RING_SIZE);
      const isNew = state.episodes[episode] === undefined;
      return {
        episodes: {
          ...state.episodes,
          [episode]: { ...existing, episode, steps: trimmed },
        },
        episodeOrder: isNew
          ? [...state.episodeOrder, episode]
          : state.episodeOrder,
        activeEpisode: state.activeEpisode ?? episode,
        hydrated: true,
      };
    }),

  seedEpisodeSummaries: (summaries) =>
    set((state) => {
      const next: Record<number, EpisodeBucket> = { ...state.episodes };
      const order = [...state.episodeOrder];
      for (const sum of summaries) {
        const existing = next[sum.episode] ?? emptyBucket(sum.episode);
        next[sum.episode] = { ...existing, episode: sum.episode, summary: sum };
        if (!state.episodes[sum.episode] && !order.includes(sum.episode)) {
          order.push(sum.episode);
        }
      }
      return { episodes: next, episodeOrder: order };
    }),
}));

export const selectActiveBucket = (
  s: ReturnType<typeof useDreamerStore.getState>,
): EpisodeBucket | null =>
  s.activeEpisode === null ? null : s.episodes[s.activeEpisode] ?? null;

export const RING_BUFFER_SIZE = RING_SIZE;

import type { StepInfo } from "@/types";
import { TURNOVER_PENALTY } from "@/lib/constants";

/** Decompose reward into log-return component and turnover penalty.
 *  log_ret = reward + 0.05 * turnover  (because reward = log_ret - 0.05 * turnover) */
export function decomposeReward(s: StepInfo): { logRet: number; penalty: number } {
  const penalty = -TURNOVER_PENALTY * s.turnover;
  const logRet = s.reward - penalty; // == s.reward + 0.05 * s.turnover
  return { logRet, penalty };
}

/** Shannon entropy in nats. Uniform-5 -> log(5) ≈ 1.6094. */
export function entropy(probs: number[]): number {
  let h = 0;
  for (const p of probs) {
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

export type EpisodeStats = {
  episode: number;
  steps: number;
  totalReward: number;
  finalEquity: number;
  maxDrawdown: number;
  meanTurnover: number;
  meanAbsLogRet: number;
};

export function computeEpisodeStats(steps: StepInfo[], episode: number): EpisodeStats {
  if (steps.length === 0) {
    return {
      episode,
      steps: 0,
      totalReward: 0,
      finalEquity: 0,
      maxDrawdown: 0,
      meanTurnover: 0,
      meanAbsLogRet: 0,
    };
  }
  let totalReward = 0;
  let peak = steps[0]!.equity;
  let maxDD = 0;
  let sumTurn = 0;
  let sumAbsLogRet = 0;
  for (const s of steps) {
    totalReward += s.reward;
    if (s.equity > peak) peak = s.equity;
    const dd = peak > 0 ? (peak - s.equity) / peak : 0;
    if (dd > maxDD) maxDD = dd;
    sumTurn += s.turnover;
    const { logRet } = decomposeReward(s);
    sumAbsLogRet += Math.abs(logRet);
  }
  return {
    episode,
    steps: steps.length,
    totalReward,
    finalEquity: steps[steps.length - 1]!.equity,
    maxDrawdown: maxDD,
    meanTurnover: sumTurn / steps.length,
    meanAbsLogRet: sumAbsLogRet / steps.length,
  };
}

export function actionHistogram(steps: StepInfo[], nActions = 5): number[] {
  const counts = new Array<number>(nActions).fill(0);
  for (const s of steps) {
    if (s.action >= 0 && s.action < nActions) counts[s.action]! += 1;
  }
  return counts;
}

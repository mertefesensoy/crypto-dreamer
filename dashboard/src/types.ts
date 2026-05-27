/**
 * Mirrors the StepInfo dataclass on the Python side, plus the fields
 * that run_random.py adds before publishing (episode, step, action_probs,
 * agent_id).
 *
 * If a field is added in Python, mirror it here. Append at the end.
 */
export type StepInfo = {
  ts: string; // ISO 8601, UTC
  price: number;
  action: number;
  target_alloc: number;
  realized_alloc: number;
  cash: number;
  btc: number;
  equity: number;
  turnover: number;
  fee_paid: number;
  reward: number;
  features_b64: string; // base64(float16, row-major, shape [16, 12])
  episode: number;
  step: number;
  action_probs: number[];
  agent_id: string;
};

export type EpisodeSummary = {
  episode: number;
  steps: number;
  total_reward: number;
  final_equity: number;
  agent_id?: string;
};

export type WsMessage =
  | { channel: "dreamer:steps"; data: StepInfo }
  | { channel: "dreamer:episodes"; data: EpisodeSummary };

export type ConnectionState = "connecting" | "open" | "closed" | "error";

// ---------------- Phase 5.4: Dream Player ----------------

export type DreamEpisode = {
  episode_id: string; // "<agent_id>|<episode_int>"
  agent_id: string;
  episode: number;
  n_steps: number;
  start_ts: string;
  end_ts: string;
  final_equity: number;
};

export type DreamRequest = {
  episode_id: string;
  start_step: number;
  action_sequence: number[]; // length 1..200, each in [0,4]
  n_samples?: number; // default 32
};

export type DreamResponse = {
  predicted_obs: number[][][]; // (n_samples, H, 15)
  predicted_rewards: number[][]; // (n_samples, H)
  real_subsequent_obs: number[][]; // (H_real, 15)
  real_subsequent_rewards: number[]; // (H_real,)
  n_steps: number;
};

export function fmtUsd(n: number, digits = 0): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtPct(n: number, digits = 1): string {
  return `${(n * 100).toFixed(digits)}%`;
}

export function fmtNum(n: number, digits = 4): string {
  return n.toFixed(digits);
}

export function fmtTime(iso: string): string {
  // Strip seconds for compact axes.
  const d = new Date(iso);
  return `${d.getUTCHours().toString().padStart(2, "0")}:${d
    .getUTCMinutes()
    .toString()
    .padStart(2, "0")}`;
}

export const ACTION_COLORS = [
  "#71717a",
  "#3f6212",
  "#65a30d",
  "#a3e635",
  "#d9f99d",
];

export const ACTION_LABELS = ["0%", "25%", "50%", "75%", "100%"];

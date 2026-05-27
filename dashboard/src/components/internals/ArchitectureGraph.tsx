import { useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  type Edge,
  type Node,
} from "reactflow";
import { useDreamerStore } from "@/store/useDreamerStore";

const NODE_BASE: Node[] = [
  {
    id: "obs",
    position: { x: 0, y: 120 },
    data: { label: "Obs · 256×12 + portfolio·3" },
    type: "input",
    style: nodeStyle("#27272a"),
  },
  {
    id: "encoder",
    position: { x: 220, y: 120 },
    data: { label: "Encoder · iTransformer" },
    style: nodeStyle("#1f1f23"),
  },
  {
    id: "rssm",
    position: { x: 460, y: 120 },
    data: { label: "RSSM · h_t, z_t" },
    style: nodeStyle("#1f1f23"),
  },
  {
    id: "actor",
    position: { x: 720, y: 40 },
    data: { label: "Actor · π(a|s)" },
    style: nodeStyle("#1f1f23"),
  },
  {
    id: "critic",
    position: { x: 720, y: 200 },
    data: { label: "Critic · IQN quantiles" },
    style: nodeStyle("#1f1f23"),
  },
  {
    id: "env",
    position: { x: 940, y: 120 },
    data: { label: "SpotBTCEnv · 5 actions" },
    type: "output",
    style: nodeStyle("#27272a"),
  },
];

const EDGE_IDS = [
  ["obs", "encoder"],
  ["encoder", "rssm"],
  ["rssm", "actor"],
  ["rssm", "critic"],
  ["actor", "env"],
  ["env", "obs"],
] as const;

function nodeStyle(bg: string): React.CSSProperties {
  return {
    background: bg,
    color: "#e4e4e7",
    border: "1px solid #3f3f46",
    borderRadius: 4,
    padding: "8px 12px",
    fontSize: 11,
    fontFamily:
      'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
    minWidth: 160,
    textAlign: "center" as const,
  };
}

const PULSE_DURATION_MS = 700;

export function ArchitectureGraph(): JSX.Element {
  const pulseTick = useDreamerStore((s) => s.pulseTick);
  const [pulsing, setPulsing] = useState(false);

  useEffect(() => {
    if (pulseTick === 0) return;
    setPulsing(true);
    const t = window.setTimeout(() => setPulsing(false), PULSE_DURATION_MS);
    return () => window.clearTimeout(t);
  }, [pulseTick]);

  const edges = useMemo<Edge[]>(
    () =>
      EDGE_IDS.map(([a, b]) => ({
        id: `${a}-${b}`,
        source: a,
        target: b,
        animated: pulsing,
        style: {
          stroke: pulsing ? "#a3e635" : "#52525b",
          strokeWidth: pulsing ? 1.6 : 1,
          transition: "stroke 200ms ease-out",
        },
      })),
    [pulsing],
  );

  return (
    <ReactFlow
      nodes={NODE_BASE}
      edges={edges}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      panOnDrag
      zoomOnScroll={false}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      proOptions={{ hideAttribution: true }}
      style={{ background: "#0a0a0a" }}
    >
      <Background color="#27272a" gap={20} size={1} />
      <Controls
        showInteractive={false}
        style={{
          background: "#18181b",
          border: "1px solid #3f3f46",
          borderRadius: 4,
        }}
      />
    </ReactFlow>
  );
}

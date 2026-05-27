import { useState } from "react";
import { Shell, type TabId } from "@/components/Shell";
import { Live } from "@/components/Live";
import { Training } from "@/components/Training";
import { Internals } from "@/components/Internals";
import { useDreamerStream } from "@/hooks/useDreamerStream";
import { useHistoryHydration } from "@/hooks/useHistoryHydration";

export function App(): JSX.Element {
  useHistoryHydration();
  useDreamerStream();
  const [tab, setTab] = useState<TabId>("live");

  return (
    <Shell active={tab} onTab={setTab}>
      {tab === "live" && <Live />}
      {tab === "training" && <Training />}
      {tab === "internals" && <Internals />}
    </Shell>
  );
}

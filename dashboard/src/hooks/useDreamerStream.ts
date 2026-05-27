import { useEffect } from "react";
import { useDreamerStore } from "@/store/useDreamerStore";
import type { WsMessage } from "@/types";

const DEFAULT_URL = "ws://127.0.0.1:8000/ws";
const RECONNECT_DELAY_MS = 1500;

/**
 * Owns a single WebSocket connection. Pushes incoming frames into the
 * Zustand store. Auto-reconnects on close.
 *
 * Mount once at the top of the app. Components subscribe to the store,
 * not to the socket.
 */
export function useDreamerStream(url: string = DEFAULT_URL): void {
  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let cancelled = false;

    const setConnection = useDreamerStore.getState().setConnection;
    const pushStep = useDreamerStore.getState().pushStep;
    const pushSummary = useDreamerStore.getState().pushSummary;

    const connect = (): void => {
      if (cancelled) return;
      setConnection("connecting");
      socket = new WebSocket(url);

      socket.onopen = () => {
        setConnection("open");
      };

      socket.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data as string) as WsMessage;
          if (msg.channel === "dreamer:steps") {
            pushStep(msg.data);
          } else if (msg.channel === "dreamer:episodes") {
            pushSummary(msg.data);
          }
        } catch {
          // Drop malformed frames silently. Keep the stream alive.
        }
      };

      socket.onerror = () => {
        setConnection("error");
      };

      socket.onclose = () => {
        setConnection("closed");
        if (cancelled) return;
        reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY_MS);
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.close();
      }
    };
  }, [url]);
}

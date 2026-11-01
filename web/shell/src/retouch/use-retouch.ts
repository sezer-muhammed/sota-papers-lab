import { useSyncExternalStore } from "react";
import { client } from "../api";
import { downloadArtifact } from "./artifact";
import { RetouchSession } from "./session";

let session: RetouchSession | undefined;

function browserStorage(): Storage | undefined {
  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}

/** The one Retouch session. Created on the page's first visit and kept for
    the life of the tab, so uploads, polls and the queue go on while you look
    at Runs or Ops. */
export function getRetouchSession(): RetouchSession {
  session ??= new RetouchSession({
    client: () => client,
    storage: browserStorage(),
    download: downloadArtifact,
    browser: true,
  });
  return session;
}

export function useRetouch() {
  const current = getRetouchSession();
  const state = useSyncExternalStore(current.subscribe, current.getSnapshot);
  return { state, session: current };
}

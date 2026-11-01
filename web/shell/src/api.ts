import { useCallback, useEffect, useRef, useState } from "react";
import { SplClient, SplError } from "@spl/sdk";

const TOKEN_KEY = "spl-token";

function readToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? undefined;
  } catch {
    return undefined;
  }
}

export function saveToken(token: string) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* not remembered in private mode */
  }
  client = new SplClient({ token });
}

/** Whether this browser holds a token (never exposes the value). */
export function hasToken(): boolean {
  return Boolean(readToken());
}

// Same origin: the host serves this build, or Vite proxies /v1 in dev.
export let client = new SplClient({ token: readToken() });

export function isUnauthorized(error: unknown) {
  return error instanceof SplError && error.status === 401;
}

/** What useLoad returns. A reload keeps the last data and error until it settles;
    when calls overlap, only the latest one settles them. */
export type Loaded<T> = {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  reload: () => Promise<void>;
};

export function useLoad<T>(load: () => Promise<T>, deps: unknown[] = []): Loaded<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(true);
  // Numbers each call, so a slower answer to an older call is dropped.
  const lastCall = useRef(0);

  const reload = useCallback(async () => {
    const call = ++lastCall.current;
    setLoading(true);
    try {
      const value = await load();
      if (call !== lastCall.current) return;
      setData(value);
      setError(undefined);
    } catch (e) {
      if (call === lastCall.current) setError(e);
    } finally {
      if (call === lastCall.current) setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { data, error, loading, reload };
}

/** Re-renders every `intervalMs` so relative times ("2 min ago") stay current. */
export function useNow(intervalMs = 60_000): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs]);

  return now;
}

/** Calls `refresh` when the tab becomes visible or the window regains focus,
    at most once per `minIntervalMs`. No polling. */
export function useRefreshOnReturn(refresh: () => void, minIntervalMs = 10_000) {
  const latest = useRef(refresh);

  useEffect(() => {
    latest.current = refresh;
  });

  useEffect(() => {
    let last = Date.now();
    const onReturn = () => {
      if (document.visibilityState !== "visible") return;
      const now = Date.now();
      if (now - last < minIntervalMs) return;
      last = now;
      latest.current();
    };
    document.addEventListener("visibilitychange", onReturn);
    window.addEventListener("focus", onReturn);
    return () => {
      document.removeEventListener("visibilitychange", onReturn);
      window.removeEventListener("focus", onReturn);
    };
  }, [minIntervalMs]);
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

/** errorMessage, with the HTTP status for API errors and plain words for a
    network failure (fetch rejects with a TypeError). */
export function describeError(error: unknown): string {
  if (error instanceof SplError) return `HTTP ${error.status}: ${error.message}`;
  if (error instanceof TypeError) return "The host did not answer.";
  return errorMessage(error);
}

import { useCallback, useEffect, useState } from "react";
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

// Same origin: the host serves this build, or Vite proxies /v1 in dev.
export let client = new SplClient({ token: readToken() });

export function isUnauthorized(error: unknown) {
  return error instanceof SplError && error.status === 401;
}

export function useLoad<T>(load: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(true);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      setData(await load());
      setError(undefined);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { data, error, loading, reload };
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

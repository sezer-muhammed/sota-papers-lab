import { useEffect, useRef, useState } from "react";
import { client, isUnauthorized, useLoad, useRefreshOnReturn } from "./api";
import { SiteFooter } from "./components/site-footer";
import { SiteHeader } from "./components/site-header";
import { TokenPrompt } from "./components/token-prompt";
import { scrollToId } from "./lib/scroll";
import { HomePage } from "./pages/home";
import { OpsPage } from "./pages/ops";
import { RetouchPage } from "./pages/retouch";
import { RunsPage } from "./pages/runs";

export type Route =
  | { page: "home"; focus?: "host" }
  | { page: "retouch"; photo?: string }
  | { page: "ops"; opId?: string }
  | { page: "runs"; runId?: string };

const decode = (s: string) => {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
};

/* "#/retouch/<photo>", "#/ops/<op>", "#/runs/<id>", "#/host" (old
   "#/overview"); "", "#/home" and anything unknown are Home. A new object on
   every call, so the same hash twice still re-runs route effects. */
function routeFromHash(hash = window.location.hash): Route {
  const [head = "", ...rest] = hash.replace(/^#\/?/, "").split("/");
  const param = rest.length ? decode(rest.join("/")) || undefined : undefined;
  if (head === "retouch") return { page: "retouch", photo: param };
  if (head === "ops") return { page: "ops", opId: param };
  if (head === "runs") return { page: "runs", runId: param };
  if (head === "host" || head === "overview") return { page: "home", focus: "host" };
  return { page: "home" };
}

const titles: Record<Route["page"], string> = {
  home: "SOTA Papers Lab",
  retouch: "Retouch · SOTA Papers Lab",
  ops: "Ops · SOTA Papers Lab",
  runs: "Runs · SOTA Papers Lab",
};

export function App() {
  const [route, setRoute] = useState<Route>(() => routeFromHash());
  const health = useLoad(() => client.health());
  // The route the page loaded with: for it the browser restores scroll and
  // focus. Compared by identity, so StrictMode's repeated effect still sees it.
  const loadRoute = useRef(route);

  useRefreshOnReturn(() => void health.reload());

  useEffect(() => {
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const first = route === loadRoute.current;
    if (window.location.hash === "#/overview") history.replaceState(null, "", "#/host");
    document.title = titles[route.page];

    if (route.page === "home" && route.focus === "host") {
      const frame = requestAnimationFrame(() => {
        scrollToId("host");
        if (!first) document.getElementById("host")?.focus({ preventScroll: true });
      });
      return () => cancelAnimationFrame(frame);
    }
    if (first) return;
    document.getElementById("main")?.focus({ preventScroll: true });
    // RunsPage scrolls a deep-linked run into view itself.
    if (!(route.page === "runs" && route.runId)) window.scrollTo({ top: 0, behavior: "instant" });
  }, [route]);

  return (
    <div className="flex min-h-screen flex-col px-3 py-3 sm:px-4">
      <SiteHeader health={health} route={route} />
      <main
        id="main"
        tabIndex={-1}
        className="mx-auto w-full max-w-[var(--layout-max-w)] flex-1 py-8 outline-none"
      >
        {isUnauthorized(health.error) ? (
          <>
            <h1 className="sr-only">SOTA Papers Lab</h1>
            <TokenPrompt />
          </>
        ) : route.page === "retouch" ? (
          <RetouchPage photoKey={route.photo} />
        ) : route.page === "ops" ? (
          <OpsPage initialOpId={route.opId} />
        ) : route.page === "runs" ? (
          <RunsPage initialRunId={route.runId} />
        ) : (
          <HomePage health={health} />
        )}
      </main>
      <SiteFooter version={health.error ? undefined : health.data?.version} />
    </div>
  );
}

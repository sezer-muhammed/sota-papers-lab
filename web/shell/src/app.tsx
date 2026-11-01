import { useEffect, useState } from "react";
import { client, isUnauthorized, useLoad } from "./api";
import { SiteHeader } from "./components/site-header";
import { TokenPrompt } from "./components/token-prompt";
import { OpsPage } from "./pages/ops";
import { OverviewPage } from "./pages/overview";
import { RunsPage } from "./pages/runs";

export type Route = "overview" | "ops" | "runs";

function routeFromHash(): Route {
  const name = window.location.hash.replace(/^#\/?/, "");
  return name === "ops" || name === "runs" ? name : "overview";
}

export function App() {
  const [route, setRoute] = useState<Route>(routeFromHash);
  const health = useLoad(() => client.health());

  useEffect(() => {
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="min-h-screen px-3 py-3 sm:px-4">
      <SiteHeader route={route} online={Boolean(health.data)} checking={health.loading} />
      <main className="mx-auto w-full max-w-[var(--layout-max-w)] py-8">
        {isUnauthorized(health.error) ? (
          <TokenPrompt />
        ) : route === "ops" ? (
          <OpsPage />
        ) : route === "runs" ? (
          <RunsPage />
        ) : (
          <OverviewPage health={health} />
        )}
      </main>
    </div>
  );
}

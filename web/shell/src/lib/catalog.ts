import type { OpDescriptor, RunRecord } from "@spl/sdk";

/** Runs Home asks for; the window behind each app's last run and run count. */
export const RUN_WINDOW = 100;
/** Rows in Home's Recent runs. */
export const RECENT_ROWS = 6;
/** App cards shown before "Show all N apps". */
export const APP_CAP = 8;
/** Op rows per app card before "+N more ops". */
export const OP_ROWS = 3;

/* An app is the op id prefix before the first dot ("device_info.list" →
   "device_info"). The host has no app endpoint; this is derived from ops. */
export type AppSummary = { id: string; ops: OpDescriptor[]; lastRun?: RunRecord; runCount: number };

const appIdOf = (opId: string) => {
  const i = opId.indexOf(".");
  return i === -1 ? opId : opId.slice(0, i);
};

/** Groups ops into apps and attaches run stats from `runs` (newest first).
    Apps with a run come first, most recently run first; the rest by id. */
export function groupApps(ops: OpDescriptor[], runs: RunRecord[]): AppSummary[] {
  const apps = new Map<string, AppSummary>();
  for (const op of ops) {
    const id = appIdOf(op.id);
    const app = apps.get(id) ?? { id, ops: [], runCount: 0 };
    app.ops.push(op);
    apps.set(id, app);
  }

  // Runs of ops that are no longer linked don't count.
  const linked = new Set(ops.map((op) => op.id));
  for (const run of runs) {
    if (!linked.has(run.op_id)) continue;
    const app = apps.get(appIdOf(run.op_id));
    if (!app) continue;
    app.runCount += 1;
    app.lastRun ??= run;
  }

  const list = [...apps.values()];
  for (const app of list) app.ops.sort((a, b) => a.id.localeCompare(b.id));
  return list.sort((a, b) => {
    if (a.lastRun && b.lastRun) return b.lastRun.created_at_ms - a.lastRun.created_at_ms;
    if (a.lastRun) return -1;
    if (b.lastRun) return 1;
    return a.id.localeCompare(b.id);
  });
}

/** "device_info" → "DI"; a one-word id → its first two letters. */
export const initials = (id: string) => {
  const [first = id, second] = id.split(/[_\-\s]+/).filter(Boolean);
  return (second ? first.charAt(0) + second.charAt(0) : first.slice(0, 2)).toUpperCase();
};

/** Case-insensitive match on the app id, its op ids and op summaries. */
export const matchesApp = (app: AppSummary, q: string) => {
  const needle = q.trim().toLowerCase();
  if (!needle) return true;
  return (
    app.id.toLowerCase().includes(needle) ||
    app.ops.some(
      (op) => op.id.toLowerCase().includes(needle) || op.summary.toLowerCase().includes(needle),
    )
  );
};

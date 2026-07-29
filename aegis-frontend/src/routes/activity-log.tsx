import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { PageHeader } from "@/components/app-shell";
import { api } from "@/lib/api";

export const Route = createFileRoute("/activity-log")({
  head: () => ({ meta: [{ title: "Activity Log — Aegis Credit" }] }),
  component: ActivityLog,
});

interface LogRow {
  run_id: string;
  timestamp: string;
  stage: string;
  summary: unknown;
  artifact_path?: string;
}

interface MergedRow extends LogRow {
  workspace: "Dev" | "Validation";
  logFile: string;
}

function summarize(summary: unknown): string {
  if (summary == null) return "-";
  if (typeof summary === "string") return summary;
  try {
    const entries = Object.entries(summary as Record<string, unknown>);
    return entries
      .slice(0, 4)
      .map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : String(v)}`)
      .join(" · ");
  } catch {
    return String(summary);
  }
}

function ActivityLog() {
  const [rows, setRows] = useState<MergedRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<MergedRow | null>(null);
  const [artifact, setArtifact] = useState<unknown>(null);
  const [artifactLoading, setArtifactLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [devRes, valRes] = await Promise.all([
          api<{ runs: LogRow[] }>("/history/dev"),
          api<{ runs: LogRow[] }>("/history/validation"),
        ]);
        if (cancelled) return;
        const merged: MergedRow[] = [
          ...(devRes?.runs ?? []).map((r) => ({ ...r, workspace: "Dev" as const, logFile: "dev_pipeline_log.csv" })),
          ...(valRes?.runs ?? []).map((r) => ({ ...r, workspace: "Validation" as const, logFile: "validation_pipeline_log.csv" })),
        ].sort((a, b) => (b.timestamp || "").localeCompare(a.timestamp || ""));
        setRows(merged);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load activity log.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleRowClick = async (row: MergedRow) => {
    setSelected(row);
    setArtifact(null);
    setArtifactLoading(true);
    try {
      const res = await api<{ artifact: unknown }>(
        `/history/artifact?log_file=${encodeURIComponent(row.logFile)}&run_id=${encodeURIComponent(row.run_id)}`,
      );
      setArtifact(res?.artifact ?? null);
    } catch {
      setArtifact(null);
    } finally {
      setArtifactLoading(false);
    }
  };

  const empty = useMemo(() => !loading && rows.length === 0, [loading, rows]);

  return (
    <div>
      <PageHeader
        title="Activity Log"
        description="Every pipeline step ever saved, across both the Model Development and Model Validation workspaces."
      />

      {error && (
        <div className="mb-4 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="overflow-hidden rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted/40 text-left text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-4 py-2.5 font-medium">Timestamp</th>
              <th className="px-4 py-2.5 font-medium">Workspace</th>
              <th className="px-4 py-2.5 font-medium">Stage</th>
              <th className="px-4 py-2.5 font-medium">Summary</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={`${row.logFile}-${row.run_id}`}
                onClick={() => handleRowClick(row)}
                className="cursor-pointer border-t border-border/70 hover:bg-muted/30"
              >
                <td className="whitespace-nowrap px-4 py-2.5 text-muted-foreground">{row.timestamp}</td>
                <td className="px-4 py-2.5">
                  <span
                    className={
                      "rounded-full px-2 py-0.5 text-xs font-medium " +
                      (row.workspace === "Dev" ? "bg-primary/10 text-primary" : "bg-secondary/60 text-secondary-foreground")
                    }
                  >
                    {row.workspace}
                  </span>
                </td>
                <td className="px-4 py-2.5 font-medium">{row.stage}</td>
                <td className="max-w-md truncate px-4 py-2.5 text-muted-foreground">{summarize(row.summary)}</td>
              </tr>
            ))}
            {loading && (
              <tr>
                <td colSpan={4} className="px-4 py-6 text-center text-muted-foreground">
                  Loading activity…
                </td>
              </tr>
            )}
            {empty && (
              <tr>
                <td colSpan={4} className="px-4 py-6 text-center text-muted-foreground">
                  No runs saved yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {selected && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          onClick={() => setSelected(null)}
        >
          <div
            className="max-h-[80vh] w-full max-w-3xl overflow-auto rounded-lg border border-border bg-card p-5 shadow-elegant"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-3 flex items-start justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold">
                  {selected.workspace} · {selected.stage}
                </h2>
                <p className="text-xs text-muted-foreground">
                  {selected.timestamp} — run {selected.run_id}
                </p>
              </div>
              <button
                onClick={() => setSelected(null)}
                className="rounded-md border border-border px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
              >
                Close
              </button>
            </div>
            {artifactLoading && <p className="text-sm text-muted-foreground">Loading full payload…</p>}
            {!artifactLoading && artifact == null && (
              <p className="text-sm text-muted-foreground">No full payload was saved for this run.</p>
            )}
            {!artifactLoading && artifact != null && (
              <pre className="whitespace-pre-wrap break-words rounded-md bg-muted/40 p-3 text-xs">
                {JSON.stringify(artifact, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

interface LatestRunResponse<T> {
  latest: {
    run_id: string;
    timestamp: string;
    summary: unknown;
    full_payload: T | null;
  } | null;
}

/**
 * Fetches the most recently saved run for a given pipeline stage so a page
 * can resume where the user left off instead of starting blank.
 *
 * Fails silently: any network/parsing error just leaves `data` as null, so
 * callers fall back to their normal empty/initial state exactly as before.
 */
export function useResumeState<T = unknown>(logFile: string, stage: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;

    let cancelled = false;

    (async () => {
      try {
        const res = await api<LatestRunResponse<T>>(
          `/history/latest?log_file=${encodeURIComponent(logFile)}&stage=${encodeURIComponent(stage)}`,
        );
        if (!cancelled && res?.latest?.full_payload) {
          setData(res.latest.full_payload);
        }
      } catch {
        // Silent — resume is a convenience, never blocks the page.
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logFile, stage]);

  return { data, loading };
}

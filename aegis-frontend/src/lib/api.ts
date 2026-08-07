/**
 * Thin HTTP client for the FastAPI backend (see `backend/api/`).
 *
 * The Aegis UI now renders from FastAPI endpoints only.
 */
const BASE_URL = import.meta.env.VITE_API_URL ?? "";
// Default to local backend during development when VITE_API_URL is not set
const EFFECTIVE_BASE = BASE_URL || "http://localhost:8000";
console.log("VITE_API_URL =", import.meta.env.VITE_API_URL);
console.log("EFFECTIVE_BASE =", EFFECTIVE_BASE);

export class ApiError extends Error {
  constructor(public status: number, message: string, public body?: unknown) {
    super(message);
  }
}

// FastAPI error responses are {"detail": "..."} (or occasionally a list of
// validation errors) — pull that out so ApiError.message is the actual
// reason ("FRED_API_KEY environment variable is not set on the server")
// instead of just "500 Internal Server Error", which told the user nothing
// about what to fix.
function extractDetail(status: number, statusText: string, body: unknown): string {
  if (body && typeof body === "object" && "detail" in (body as any)) {
    const detail = (body as any).detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      return detail.map((d) => (typeof d === "string" ? d : d?.msg ?? JSON.stringify(d))).join("; ");
    }
  }
  return `${status} ${statusText}`;
}

export async function api<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${EFFECTIVE_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });
  const text = await res.text();
  const body = text ? safeJson(text) : undefined;
  if (!res.ok) {
    throw new ApiError(res.status, extractDetail(res.status, res.statusText, body), body);
  }
  return body as T;
}

/**
 * Upload a FormData payload. Does not override Content-Type so the browser
 * can set multipart boundaries correctly.
 */
export async function formUpload<T = unknown>(path: string, form: FormData): Promise<T> {
  const url = `${EFFECTIVE_BASE}${path}`;
  console.log("formUpload: POST", url);
  try {
    const res = await fetch(url, {
      method: "POST",
      body: form,
    });
    console.log("formUpload: response status", res.status);
    const text = await res.text();
    const body = text ? safeJson(text) : undefined;
    if (!res.ok) {
      console.error("formUpload: response error", res.status, body);
      throw new ApiError(res.status, extractDetail(res.status, res.statusText, body), body);
    }
    return body as T;
  } catch (err) {
    console.error("formUpload: fetch error", err);
    throw err;
  }
}

export const apiUrl = (path: string) => `${EFFECTIVE_BASE}${path}`;

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
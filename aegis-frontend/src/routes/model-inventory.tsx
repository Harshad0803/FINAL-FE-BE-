import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { PageHeader } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { ArrowRight, Database, Download, FileCheck, FileText, History, Plus, Search, SlidersHorizontal, ShieldCheck, Sparkles } from "lucide-react";

type InventoryModel = {
  model_id: string;
  model_name: string;
  model_type: string;
  algorithm?: string;
  business_purpose: string;
  model_owner: string;
  business_unit: string;
  model_version: string;
  regulatory_framework: string;
  status: string;
  development_date: string;
  implementation_date?: string;
  last_validation_date?: string;
  next_validation_due?: string;
  model_risk_rating: string;
  approval_status: string;
  reviewer: string;
  created_at: string;
  updated_at: string;
  documentation_url?: string;
  documentation_path?: string;
  document_url?: string;
  document_path?: string;
};

type InventoryDataSource = {
  data_source_id: string;
  model_id: string;
  file_name: string;
  source_type: string;
  purpose: string;
  uploaded_by: string;
  uploaded_at: string;
  record_count?: number;
  column_count?: number;
  target_variable?: string;
  storage_reference?: string;
};

type InventoryValidation = {
  model_id: string;
  data_validation_status: string;
  conceptual_soundness_status: string;
  backtesting_status: string;
  overall_status: string;
  findings_count: number;
  last_validation?: string | null;
};

type InventoryHistoryEntry = {
  model_id: string;
  event: string;
  description: string;
  timestamp: string;
  user: string;
  validation_run_id?: string;
};

type InventoryTab = "development" | "validation";

type InventoryPayload = {
  development: InventoryModel[];
  validation: InventoryModel[];
  models: InventoryModel[];
  data_sources: InventoryDataSource[];
  history: InventoryHistoryEntry[];
};

export const Route = createFileRoute("/model-inventory")({
  head: () => ({ meta: [{ title: "Model Inventory — Aegis Credit" }] }),
  component: ModelInventory,
});

function normalizeInventoryPayload(payload: InventoryPayload): InventoryPayload {
  return {
    ...payload,
    models: payload.models.map((model) => ({
      ...model,
      algorithm: typeof model.algorithm === "string" && model.algorithm.trim() ? model.algorithm.trim() : undefined,
    })),
  };
}

function displayAlgorithm(model: InventoryModel): string {
  return typeof model.algorithm === "string" && model.algorithm.trim() ? model.algorithm.trim() : "—";
}

function getDocumentationReference(model: InventoryModel): string | null {
  const candidates = [
    model.documentation_url,
    model.documentation_path,
    model.document_url,
    model.document_path,
  ];

  for (const candidate of candidates) {
    const value = typeof candidate === "string" ? candidate.trim() : "";
    if (value) {
      return value;
    }
  }

  return null;
}

function resolveDocumentationHref(reference: string | null): string | null {
  if (!reference) return null;
  if (/^https?:\/\//i.test(reference) || reference.startsWith("data:")) {
    return reference;
  }
  const apiBase = (import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8001").replace(/\/$/, "");
  return `${apiBase}/model-inventory/documentation?path=${encodeURIComponent(reference)}`;
}

function ModelInventory() {
  const [inventory, setInventory] = useState<InventoryPayload | null>(null);
  const [activeTab, setActiveTab] = useState<InventoryTab>("development");
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [buFilter, setBuFilter] = useState<string | null>(null);
  const [form, setForm] = useState({ model_name: "", model_owner: "", business_unit: "Retail Credit", model_version: "1.0", regulatory_framework: "Internal", model_type: "Probability of Default" });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadInventory = async () => {
    try {
      const payload = await api<InventoryPayload>("/model-inventory");
      setInventory(normalizeInventoryPayload(payload));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load inventory");
    }
  };

  useEffect(() => { void loadInventory(); }, []);

  const currentRecords = useMemo(() => {
    if (!inventory) return [] as InventoryModel[];
    return activeTab === "development" ? (inventory.development ?? []) : (inventory.validation ?? []);
  }, [inventory, activeTab]);

  const selectedModel = useMemo(() => currentRecords[0] ?? null, [currentRecords]);

  const filteredModels = useMemo(() => {
    if (!currentRecords) return [] as InventoryModel[];
    return currentRecords.filter((m) => {
      const q = search.trim().toLowerCase();
      if (q) {
        const inText = [m.model_name, m.model_type, displayAlgorithm(m), m.model_owner, m.regulatory_framework].join(" ").toLowerCase();
        if (!inText.includes(q)) return false;
      }
      if (statusFilter && m.status !== statusFilter) return false;
      if (buFilter && m.business_unit !== buFilter) return false;
      return true;
    });
  }, [currentRecords, search, statusFilter, buFilter]);

  const summary = useMemo(() => {
    const totals = { total: 0, validated: 0, inReview: 0, highPriority: 0 };
    if (!inventory) return totals;
    const source = activeTab === "development" ? (inventory.development ?? []) : (inventory.validation ?? []);
    totals.total = source.length;
    for (const m of source) {
      if (m.status === "Validated") totals.validated += 1;
      if (m.status === "In Review") totals.inReview += 1;
      if (m.model_risk_rating === "High" || m.model_risk_rating === "Critical") totals.highPriority += 1;
    }
    return totals;
  }, [inventory, activeTab]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const created = await api<InventoryModel>("/model-inventory/models", {
        method: "POST",
        body: JSON.stringify({
          model_name: form.model_name,
          model_owner: form.model_owner,
          business_unit: form.business_unit,
          model_version: form.model_version,
          regulatory_framework: form.regulatory_framework,
          model_type: form.model_type,
          status: "Draft",
        }),
      });
      setForm({ model_name: "", model_owner: "", business_unit: "Retail Credit", model_version: "1.0", regulatory_framework: "Internal", model_type: "Probability of Default" });
      await loadInventory();
      if (created.model_id) {
        const next = inventory?.models ?? [];
        const modelEntry = [created, ...next].find((item) => item.model_id === created.model_id);
        if (modelEntry) {
          setInventory((current) => current ? { ...current, models: [modelEntry, ...current.models] } : current);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create model");
    } finally {
      setSubmitting(false);
    }
  };

  const exportInventory = async () => {
    const res = await fetch(`${import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8001"}/model-inventory/export`);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || "Export failed");
    }
    const blob = await res.blob();
    if (!blob.size) {
      throw new Error("Export returned an empty file");
    }
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "model_inventory.xlsx";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  return (
      <div className="space-y-6">
      <PageHeader
        title="Model Inventory"
        description="Register and track models, data sources, validation status, and historical updates for consultant demos and governance reviews."
        actions={
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => void exportInventory().catch(() => setError("Excel export failed"))}>
              <Download className="mr-2 h-4 w-4" /> Export Excel
            </Button>
            <Button onClick={() => window.location.assign("/validation/intake")}>
              Open intake <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        }
      />

      {error ? <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">{error}</div> : null}

      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="space-y-6">
          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="grid grid-cols-4 gap-4">
              <div className="col-span-4 md:col-span-1">
                <div className="text-xs font-medium text-muted-foreground">Total models</div>
                <div className="mt-1 text-2xl font-semibold">{summary.total}</div>
              </div>
              <div className="col-span-4 md:col-span-1">
                <div className="text-xs font-medium text-muted-foreground">Validated</div>
                <div className="mt-1 text-2xl font-semibold">{summary.validated}</div>
              </div>
              <div className="col-span-4 md:col-span-1">
                <div className="text-xs font-medium text-muted-foreground">In Review</div>
                <div className="mt-1 text-2xl font-semibold">{summary.inReview}</div>
              </div>
              <div className="col-span-4 md:col-span-1">
                <div className="text-xs font-medium text-muted-foreground">High Priority</div>
                <div className="mt-1 text-2xl font-semibold">{summary.highPriority}</div>
              </div>
            </div>
            <div className="mt-4 flex gap-2">
              <Button variant={activeTab === "development" ? "default" : "outline"} onClick={() => setActiveTab("development")}>Development</Button>
              <Button variant={activeTab === "validation" ? "default" : "outline"} onClick={() => setActiveTab("validation")}>Validation</Button>
            </div>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
              <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search models, type, owner" className="col-span-1 rounded-lg border border-input px-3 py-2 text-sm" />
              <select value={statusFilter ?? ""} onChange={(e) => setStatusFilter(e.target.value || null)} className="rounded-lg border border-input px-3 py-2 text-sm">
                <option value="">All statuses</option>
                {inventory && Array.from(new Set(inventory.models.map((m) => m.status))).map((s) => (<option key={s} value={s}>{s}</option>))}
              </select>
              <select value={buFilter ?? ""} onChange={(e) => setBuFilter(e.target.value || null)} className="rounded-lg border border-input px-3 py-2 text-sm">
                <option value="">All business units</option>
                {inventory && Array.from(new Set(inventory.models.map((m) => m.business_unit))).map((b) => (<option key={b} value={b}>{b}</option>))}
              </select>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-muted-foreground" />
              <h2 className="text-lg font-semibold">Register a model</h2>
            </div>
            <form onSubmit={handleCreate} className="mt-4 grid gap-4 md:grid-cols-2">
              <label className="text-sm font-medium">Model name
                <input required value={form.model_name} onChange={(e) => setForm((f) => ({ ...f, model_name: e.target.value }))} className="mt-1 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm" />
              </label>
              <label className="text-sm font-medium">Model owner
                <input value={form.model_owner} onChange={(e) => setForm((f) => ({ ...f, model_owner: e.target.value }))} className="mt-1 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm" />
              </label>
              <label className="text-sm font-medium">Business unit
                <input value={form.business_unit} onChange={(e) => setForm((f) => ({ ...f, business_unit: e.target.value }))} className="mt-1 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm" />
              </label>
              <label className="text-sm font-medium">Model version
                <input value={form.model_version} onChange={(e) => setForm((f) => ({ ...f, model_version: e.target.value }))} className="mt-1 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm" />
              </label>
              <label className="text-sm font-medium">Regulatory framework
                <input value={form.regulatory_framework} onChange={(e) => setForm((f) => ({ ...f, regulatory_framework: e.target.value }))} className="mt-1 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm" />
              </label>
              <label className="text-sm font-medium">Model type
                <input value={form.model_type} onChange={(e) => setForm((f) => ({ ...f, model_type: e.target.value }))} className="mt-1 w-full rounded-lg border border-input bg-background px-3 py-2 text-sm" />
              </label>
              <div className="md:col-span-2 flex justify-end">
                <Button type="submit" disabled={submitting}>{submitting ? "Saving…" : <><Plus className="mr-2 h-4 w-4" />Save model</>}</Button>
              </div>
            </form>
          </div>

          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-muted-foreground" />
                <h2 className="text-lg font-semibold">Inventory snapshot</h2>
              </div>
            </div>
            <div className="mt-4">
              <div className="mb-2 text-sm text-muted-foreground">Showing {filteredModels.length} of {currentRecords.length} {activeTab} models</div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-muted-foreground">
                      <th className="px-3 py-2">Model Name</th>
                      <th className="px-3 py-2">Model ID</th>
                      <th className="px-3 py-2">Type</th>
                      <th className="px-3 py-2">Version</th>
                      <th className="px-3 py-2">Algorithm</th>
                      <th className="px-3 py-2">Framework</th>
                      <th className="px-3 py-2">Owner</th>
                      <th className="px-3 py-2">Business Unit</th>
                      <th className="px-3 py-2">Status</th>
                      <th className="px-3 py-2">Validation</th>
                      <th className="px-3 py-2">Last Updated</th>
                      <th className="px-3 py-2">Documentation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredModels.map((m) => {
                      const documentationReference = getDocumentationReference(m);
                      const documentationHref = resolveDocumentationHref(documentationReference);

                      return (
                        <tr key={m.model_id} className="border-t border-border bg-background align-top">
                          <td className="px-3 py-2">
                            <div className="font-medium">{m.model_name}</div>
                          </td>
                          <td className="px-3 py-2 text-xs text-muted-foreground">{m.model_id}</td>
                          <td className="px-3 py-2">{m.model_type}</td>
                          <td className="px-3 py-2">{m.model_version}</td>
                          <td className="px-3 py-2">{displayAlgorithm(m)}</td>
                          <td className="px-3 py-2">{m.regulatory_framework}</td>
                          <td className="px-3 py-2">{m.model_owner}</td>
                          <td className="px-3 py-2">{m.business_unit}</td>
                          <td className="px-3 py-2"><span className="rounded-full bg-primary-soft px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-primary">{m.status}</span></td>
                          <td className="px-3 py-2">{m.status}</td>
                          <td className="px-3 py-2">{new Date(m.updated_at).toLocaleString()}</td>
                          <td className="px-3 py-2">
                            {documentationHref ? (
                              <a className="text-primary underline" href={documentationHref} target="_blank" rel="noreferrer">Open</a>
                            ) : (
                              <span className="text-muted-foreground">Not available</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-muted-foreground" />
              <h2 className="text-lg font-semibold">Data sources</h2>
            </div>
            <div className="mt-4 space-y-3">
              {inventory?.data_sources.length ? inventory.data_sources.map((source) => (
                <div key={source.data_source_id} className="rounded-lg border border-border bg-background p-3 text-sm">
                  <div className="font-medium">{source.file_name}</div>
                  <div className="mt-1 text-xs text-muted-foreground">{source.purpose} · {source.source_type}</div>
                  <div className="mt-1 text-xs text-muted-foreground">Uploaded by {source.uploaded_by} on {source.uploaded_at}</div>
                </div>
              )) : <div className="text-sm text-muted-foreground">No data sources have been linked yet.</div>}
            </div>
          </div>

          <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
            <div className="flex items-center gap-2">
              <History className="h-5 w-5 text-muted-foreground" />
              <h2 className="text-lg font-semibold">Activity</h2>
            </div>
            <div className="mt-4 space-y-3">
              {inventory?.history.slice(0, 8).map((entry, index) => (
                <div key={`${entry.timestamp}-${index}`} className="rounded-lg border border-border bg-background p-3 text-sm">
                  <div className="font-medium">{entry.event}</div>
                  <div className="mt-1 text-xs text-muted-foreground">{entry.description}</div>
                  <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-wider text-muted-foreground">
                    <span>{entry.timestamp}</span>
                    {entry.validation_run_id ? <span className="rounded-full bg-primary-soft px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-primary">{entry.validation_run_id}</span> : null}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {selectedModel ? (
            <div className="rounded-xl border border-border bg-card p-6 shadow-elegant">
              <div className="text-sm font-semibold">Selected focus model</div>
              <div className="mt-2 text-lg font-semibold">{selectedModel.model_name}</div>
              <div className="mt-2 text-sm text-muted-foreground">{selectedModel.business_purpose}</div>
              <div className="mt-3 text-sm text-muted-foreground">{activeTab === "development" ? "Development status" : "Validation status"}: {selectedModel.status}</div>
              <div className="mt-2 text-sm text-muted-foreground">Algorithm: {displayAlgorithm(selectedModel)}</div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

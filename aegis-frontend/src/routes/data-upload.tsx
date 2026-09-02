import { createFileRoute, useNavigate } from "@tanstack/react-router";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Database,
  Download,
  FileSpreadsheet,
  Globe,
  Info,
  Landmark,
  Link2,
  Loader,
  RefreshCw,
  Table2,
  TrendingUp,
  Upload,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import { formUpload } from "@/lib/api";
import { useDataset } from "@/lib/app-context";
import AnimatedNumber from "@/components/animated-number";

export const Route = createFileRoute("/data-upload")({
  head: () => ({ meta: [{ title: "Data Upload — Aegis Credit" }] }),
  component: DataUpload,
});

type TableInfo = {
  table: string;
  row_count: number;
  columns: { name: string; dtype: string; unique_ratio_sample: number; sample_values: string[] }[];
};

type JoinCandidate = {
  left_table: string;
  left_column: string;
  right_table: string;
  right_column: string;
  confidence: number;
  cardinality: string;
  reasons: string[];
};

type PrimaryKeyInfo = {
  table: string;
  column: string;
  confidence: number;
  reasons: string[];
};

type ConfirmedJoin = { right_table: string; left_key: string; right_key: string };

type MacroDateCandidate = {
  column: string;
  is_preferred: boolean;
};

type SourceStatus = "connected" | "pending" | "warning";

type FlowNode = { key: string; label: string; sub: string; state: SourceStatus };

async function readCsvHeader(file: File): Promise<string[]> {
  const text = await file.slice(0, 8192).text();
  const firstLine = text.split(/\r?\n/)[0] ?? "";
  return firstLine.split(",").map((c) => c.trim().replace(/^"|"$/g, "")).filter(Boolean);
}

function bestCandidate(candidates: JoinCandidate[], tableA: string, tableB: string): JoinCandidate | null {
  const found = candidates.find(
    (c) => (c.left_table === tableA && c.right_table === tableB) || (c.left_table === tableB && c.right_table === tableA),
  );
  if (!found) return null;
  // Normalize so left = tableA, right = tableB regardless of which order discovery found it in.
  return found.left_table === tableA
    ? found
    : {
        left_table: tableA, left_column: found.right_column, right_table: tableB, right_column: found.left_column,
        confidence: found.confidence, cardinality: found.cardinality, reasons: found.reasons,
      };
}

const SESSION_STORAGE_DATA_UPLOAD_KEY = "aegis_data_upload_active";

function DataUpload() {

  const { setUploadResult, profile } = useDataset();
  const navigate = useNavigate();

  const customerInputRef = useRef<HTMLInputElement>(null);
  const dbInputRef = useRef<HTMLInputElement>(null);

  const [customerFile, setCustomerFile] = useState<File | null>(null);
  const [customerColumns, setCustomerColumns] = useState<string[]>([]);
  const [hasExplicitDataset, setHasExplicitDataset] = useState(false);

  const [dbFile, setDbFile] = useState<File | null>(null);
  const [dbTables, setDbTables] = useState<TableInfo[] | null>(null);
  const [dbLoading, setDbLoading] = useState(false);
  const [dbError, setDbError] = useState<string | null>(null);

  const [loanTable, setLoanTable] = useState<string>("");
  const [collateralTable, setCollateralTable] = useState<string>("");

  const setExplicitDatasetActive = (active: boolean) => {
    setHasExplicitDataset(active);
    if (typeof window !== "undefined") {
      try {
        window.sessionStorage.setItem(SESSION_STORAGE_DATA_UPLOAD_KEY, active ? "1" : "0");
      } catch {
        // Ignore sessionStorage failures.
      }
    }
  };

  useEffect(() => {
    if (typeof window === "undefined") return;
    let active = false;
    try {
      const persisted = window.sessionStorage.getItem(SESSION_STORAGE_DATA_UPLOAD_KEY);
      active = persisted === "1";
    } catch {
      active = false;
    }
    setHasExplicitDataset(active);
  }, []);

  useEffect(() => {
    if (!profile || hasExplicitDataset) return;
    setUploadResult(null, null);
    setExplicitDatasetActive(false);
  }, [profile, hasExplicitDataset, setUploadResult]);

  // ── Macroeconomic data (FRED) ───────────────────────────────────────────────
  const [macroCandidates, setMacroCandidates] = useState<MacroDateCandidate[]>([]);
  const [macroCandidatesLoading, setMacroCandidatesLoading] = useState(false);
  const [selectedMacroDateCol, setSelectedMacroDateCol] = useState<string>("");
  const [macroColumns, setMacroColumns] = useState<string[]>([]);
  const [macroDateColUsed, setMacroDateColUsed] = useState<string | null>(null);
  const [macroLoading, setMacroLoading] = useState(false);
  const [macroError, setMacroError] = useState<string | null>(null);
  // The macro fetch replaces customerFile with an augmented version — keep the
  // pristine original around so "re-fetch with a different date column" never
  // stacks a second set of macro_* columns onto an already-augmented file.
  const originalCustomerFileRef = useRef<File | null>(null);

  const [candidates, setCandidates] = useState<JoinCandidate[]>([]);
  const [primaryKeys, setPrimaryKeys] = useState<PrimaryKeyInfo[]>([]);
  const [relLoading, setRelLoading] = useState(false);
  const [relError, setRelError] = useState<string | null>(null);

  const [customerLoanJoin, setCustomerLoanJoin] = useState<{ left: string; right: string } | null>(null);
  const [loanCollateralJoin, setLoanCollateralJoin] = useState<{ left: string; right: string } | null>(null);

  const [integrating, setIntegrating] = useState(false);
  const [integrateError, setIntegrateError] = useState<string | null>(null);
  const [report, setReport] = useState<any | null>(null);

  const loanColumns = useMemo(() => dbTables?.find((t) => t.table === loanTable)?.columns.map((c) => c.name) ?? [], [dbTables, loanTable]);
  const collateralColumns = useMemo(() => dbTables?.find((t) => t.table === collateralTable)?.columns.map((c) => c.name) ?? [], [dbTables, collateralTable]);

  const onCustomerFileChosen = async (f: File | null) => {
    if (!f) return;
    setCustomerFile(f);
    setReport(null);
    originalCustomerFileRef.current = f;
    setMacroColumns([]);
    setMacroDateColUsed(null);
    setMacroError(null);
    setHasExplicitDataset(true);
    try {
      setCustomerColumns(await readCsvHeader(f));
    } catch {
      setCustomerColumns([]);
    }
    if (dbTables) void discoverRelationships(f, dbFile, loanTable, collateralTable);
  };

  // Fetch date-column candidates once a customer file is available. FRED
  // needs a date to know which time period's economic data to attach to
  // each record — this is now fully automatic (same detect_macro_date_col()
  // heuristic the backend uses), so there's no manual column picker in the
  // UI. If nothing in the dataset looks like a real date column, macroError
  // is set and the fetch button stays disabled.
  useEffect(() => {
    if (!customerFile) return;
    setMacroCandidatesLoading(true);
    setMacroError(null);
    (async () => {
      try {
        const form = new FormData();
        form.append("file", customerFile);
        const res = await formUpload<{ candidates: MacroDateCandidate[]; default_date_col: string | null }>(
          "/data/macro/date-columns",
          form,
        );
        const candidateList = res.candidates ?? [];
        setMacroCandidates(candidateList);
        const bestGuess =
          res.default_date_col
          ?? candidateList.find((c) => c.is_preferred)?.column
          ?? candidateList[0]?.column
          ?? "";
        setSelectedMacroDateCol(bestGuess);
        if (!bestGuess) {
          setMacroError("No origination/observation date column could be auto-detected in this dataset — macro data can't be attached.");
        }
      } catch (err) {
        setMacroCandidates([]);
        setSelectedMacroDateCol("");
        setMacroError(
          err instanceof Error
            ? `Could not detect a date column: ${err.message}`
            : "Could not auto-detect a date column for this dataset.",
        );
      } finally {
        setMacroCandidatesLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [customerFile]);

  const fetchMacroFeatures = async () => {
    const baseFile = originalCustomerFileRef.current ?? customerFile;
    if (!baseFile || !selectedMacroDateCol) return;
    try {
      setMacroLoading(true);
      setMacroError(null);
      const form = new FormData();
      form.append("file", baseFile);
      form.append("date_col", selectedMacroDateCol);
      const res = await formUpload<{
        macro_columns: string[];
        date_col_used: string;
        csv_with_macro: string;
      }>("/data/macro/fetch", form);
      setMacroColumns(res.macro_columns ?? []);
      setMacroDateColUsed(res.date_col_used ?? selectedMacroDateCol);

      // Carry the macro-augmented dataset forward as the working customer
      // file, so it flows into the join/integration step below with the
      // macro columns already attached.
      const macroBlob = new Blob([res.csv_with_macro], { type: "text/csv" });
      const macroFile = new File([macroBlob], baseFile.name, { type: "text/csv" });
      setCustomerFile(macroFile);
      setCustomerColumns(await readCsvHeader(macroFile));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to fetch macroeconomic data";
      setMacroError(message);
    } finally {
      setMacroLoading(false);
    }
  };

  const handleReFetchMacro = () => {
    setMacroColumns([]);
    setMacroDateColUsed(null);
    setMacroError(null);
    // Revert to the pristine (pre-macro) customer file so the next fetch
    // attaches macro columns fresh instead of stacking onto the last result.
    if (originalCustomerFileRef.current) {
      setCustomerFile(originalCustomerFileRef.current);
      void readCsvHeader(originalCustomerFileRef.current).then(setCustomerColumns).catch(() => setCustomerColumns([]));
    }
  };

  const onDbFileChosen = async (f: File | null) => {
    if (!f) return;
    setDbFile(f);
    setDbError(null);
    setDbLoading(true);
    setReport(null);
    setHasExplicitDataset(true);
    try {
      const form = new FormData();
      form.append("db_file", f);
      const resp = await formUpload<{ tables: TableInfo[] }>("/data/integration/sqlite/inspect", form);
      setDbTables(resp.tables);
      const guessLoan = resp.tables.find((t) => /loan/i.test(t.table))?.table ?? resp.tables[0]?.table ?? "";
      const guessCollateral = resp.tables.find((t) => /collateral/i.test(t.table))?.table ?? "";
      setLoanTable(guessLoan);
      setCollateralTable(guessCollateral);
      if (customerFile) void discoverRelationships(customerFile, f, guessLoan, guessCollateral);
    } catch (error: any) {
      setDbTables(null);
      setDbError(error?.message ?? "Failed to read the SQLite database.");
    } finally {
      setDbLoading(false);
    }
  };

  const discoverRelationships = async (custFile: File | null, database: File | null, loan: string, collateral: string) => {
    const selected = [loan, collateral].filter(Boolean);
    if (!custFile || !database || selected.length === 0) return;
    setRelLoading(true);
    setRelError(null);
    try {
      const form = new FormData();
      form.append("customer_file", custFile);
      form.append("db_file", database);
      form.append("selected_tables", selected.join(","));
      const resp = await formUpload<{ primary_keys: PrimaryKeyInfo[]; candidates: JoinCandidate[] }>("/data/integration/relationships", form);
      setCandidates(resp.candidates);
      setPrimaryKeys(resp.primary_keys ?? []);

      const cl = loan ? bestCandidate(resp.candidates, "customer", loan) : null;
      setCustomerLoanJoin(cl ? { left: cl.left_column, right: cl.right_column } : null);

      const lc = loan && collateral ? bestCandidate(resp.candidates, loan, collateral) : null;
      setLoanCollateralJoin(lc ? { left: lc.left_column, right: lc.right_column } : null);
    } catch (error: any) {
      setRelError(error?.message ?? "Failed to discover relationships between sources.");
      setCandidates([]);
      setPrimaryKeys([]);
    } finally {
      setRelLoading(false);
    }
  };

  const onLoanTableChange = (t: string) => {
    setLoanTable(t);
    if (customerFile && dbFile) void discoverRelationships(customerFile, dbFile, t, collateralTable);
  };
  const onCollateralTableChange = (t: string) => {
    setCollateralTable(t);
    if (customerFile && dbFile) void discoverRelationships(customerFile, dbFile, loanTable, t);
  };

  const canIntegrate = Boolean(customerFile) && (!loanTable || Boolean(customerLoanJoin)) && (!collateralTable || Boolean(loanCollateralJoin));

  const runIntegration = async () => {
    if (!customerFile) return;
    setIntegrating(true);
    setIntegrateError(null);
    try {
      const joinSpecs: ConfirmedJoin[] = [];
      if (loanTable && customerLoanJoin) {
        joinSpecs.push({ right_table: loanTable, left_key: customerLoanJoin.left, right_key: customerLoanJoin.right });
      }
      if (collateralTable && loanCollateralJoin) {
        joinSpecs.push({ right_table: collateralTable, left_key: loanCollateralJoin.left, right_key: loanCollateralJoin.right });
      }

      const form = new FormData();
      form.append("customer_file", customerFile);
      if (dbFile) form.append("db_file", dbFile);
      if (loanTable) form.append("loan_table", loanTable);
      if (collateralTable) form.append("collateral_table", collateralTable);
      form.append("join_specs_json", JSON.stringify(joinSpecs));
      // Macro columns (if any) are already attached to customerFile by
      // fetchMacroFeatures above, so no fetch_macro/date-column params are
      // sent here — integration just carries them through like any other
      // customer-file column.

      const resp = await formUpload<any>("/data/integration/run", form);
      setReport(resp);

      const datasetName = resp?.dataset_name ?? "integrated_dataset.csv";
      const resolvedFile = typeof resp?.csv_text === "string"
        ? new File([resp.csv_text], datasetName.endsWith(".csv") ? datasetName : `${datasetName}.csv`, { type: "text/csv" })
        : null;
      setUploadResult(resolvedFile, resp as any);
      setExplicitDatasetActive(true);
    } catch (error: any) {
      setIntegrateError(error?.message ?? "Integration failed.");
    } finally {
      setIntegrating(false);
    }
  };

  const downloadReport = () => {
    if (!report?.integration_report) return;
    const blob = new Blob([JSON.stringify(report.integration_report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "integration_report.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const integrationReport = report?.integration_report;

  // ── Presentation-only derived values (all from state already above — no new fetches) ──
  const sourcesConnectedCount = [Boolean(customerFile), Boolean(dbFile), macroColumns.length > 0].filter(Boolean).length;

  const integrationSourceByName = (name: string): { name: string; rows: number; columns: number } | null =>
    integrationReport?.sources?.find((s: any) => s.name === name) ?? null;
  const customerSourceInfo = integrationSourceByName("customer");

  const loanTableInfo = dbTables?.find((t) => t.table === loanTable) ?? null;
  const collateralTableInfo = dbTables?.find((t) => t.table === collateralTable) ?? null;
  const loanSourceInfo = loanTable ? integrationSourceByName(loanTable) : null;
  const collateralSourceInfo = collateralTable ? integrationSourceByName(collateralTable) : null;

  const customerLoanCandidate = loanTable ? bestCandidate(candidates, "customer", loanTable) : null;
  const loanCollateralCandidate = loanTable && collateralTable ? bestCandidate(candidates, loanTable, collateralTable) : null;

  const activeJoinConfidences = [
    loanTable && customerLoanJoin ? customerLoanCandidate?.confidence : null,
    loanTable && collateralTable && loanCollateralJoin ? loanCollateralCandidate?.confidence : null,
  ].filter((c): c is number => typeof c === "number");
  const joinHealthPct = activeJoinConfidences.length
    ? Math.round((activeJoinConfidences.reduce((a, b) => a + b, 0) / activeJoinConfidences.length) * 100)
    : null;

  const flowNodes: FlowNode[] = [
    {
      key: "customer",
      label: "Customer CSV",
      sub: customerFile
        ? `${customerColumns.length} columns${customerSourceInfo ? ` · ${customerSourceInfo.rows.toLocaleString()} rows` : ""}`
        : "Not uploaded",
      state: customerFile ? "connected" : "pending",
    },
  ];
  if (loanTable) {
    flowNodes.push({
      key: "loan",
      label: loanTable,
      sub: loanSourceInfo
        ? `${loanSourceInfo.rows.toLocaleString()} rows · ${loanSourceInfo.columns} cols`
        : loanTableInfo
        ? `${loanTableInfo.row_count.toLocaleString()} rows · ${loanColumns.length} cols`
        : "Awaiting database",
      state: dbFile && loanTable ? "connected" : "pending",
    });
  }
  if (collateralTable) {
    flowNodes.push({
      key: "collateral",
      label: collateralTable,
      sub: collateralSourceInfo
        ? `${collateralSourceInfo.rows.toLocaleString()} rows · ${collateralSourceInfo.columns} cols`
        : collateralTableInfo
        ? `${collateralTableInfo.row_count.toLocaleString()} rows · ${collateralColumns.length} cols`
        : "Awaiting database",
      state: dbFile && collateralTable ? "connected" : "pending",
    });
  }
  if (macroColumns.length > 0) {
    flowNodes.push({
      key: "macro",
      label: "Macro Data",
      sub: macroDateColUsed ? `Aligned on ${macroDateColUsed}` : `${macroColumns.length} series attached`,
      state: "connected",
    });
  }
  flowNodes.push({
    key: "integrated",
    label: "Integrated Dataset",
    sub: integrationReport
      ? `${integrationReport.rows_after.toLocaleString()} rows · ${integrationReport.columns_after} cols`
      : integrating
      ? "Integrating…"
      : "Not yet integrated",
    state: integrationReport ? (integrationReport.warnings?.length ? "warning" : "connected") : "pending",
  });

  const pipelineSteps: { label: string; sub: string; status: "done" | "active" | "pending" }[] = [
    { label: "SOURCE", sub: `${sourcesConnectedCount} connected`, status: customerFile ? "done" : "pending" },
    { label: "CONNECTED", sub: canIntegrate ? "Validated" : "Pending", status: canIntegrate ? "done" : "pending" },
    {
      label: "INTEGRATED",
      sub: integrationReport ? `${integrationReport.rows_after.toLocaleString()} rows` : integrating ? "Running…" : "Pending",
      status: integrationReport ? "done" : integrating ? "active" : "pending",
    },
    { label: "PROFILED", sub: profile ? "Complete" : "Pending", status: profile ? "done" : "pending" },
    { label: "READY FOR MODELLING", sub: "", status: "pending" },
  ];

  const isReady = Boolean(hasExplicitDataset && profile);

  return (
    <div className="space-y-4">
      {/* Hero */}
      <div className="rounded-2xl border border-slate-200 bg-gradient-to-r from-slate-900 via-sky-900 to-blue-800 p-6 text-white shadow-[0_16px_36px_rgba(15,23,42,0.16)]">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white/10">
              <Link2 className="h-5 w-5 text-sky-200" />
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-[0.2em] text-sky-200">Data Integration</div>
              <h1 className="mt-2 text-lg font-semibold leading-tight">Connect a Data Source</h1>
              <p className="mt-1 max-w-lg text-[13px] text-slate-200">
                Bring customer, loan, collateral, and macroeconomic data into the modelling workflow. Sources are joined
                automatically and validated before integration.
              </p>
            </div>
          </div>
          <div className="flex flex-shrink-0 flex-wrap items-center gap-2">
            <HeroPill tone={sourcesConnectedCount > 0 ? "emerald" : "neutral"}>
              {sourcesConnectedCount > 0 && <CheckCircle2 className="h-3 w-3" />}
              {sourcesConnectedCount} SOURCE{sourcesConnectedCount === 1 ? "" : "S"} CONNECTED
            </HeroPill>
            <HeroPill tone={integrationReport ? "emerald" : canIntegrate ? "amber" : "neutral"}>
              {integrationReport ? "INTEGRATION COMPLETE" : canIntegrate ? "READY TO INTEGRATE" : "SOURCES INCOMPLETE"}
            </HeroPill>
          </div>
        </div>
      </div>

      {/* Source overview */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <SourceOverviewCard
          icon={<FileSpreadsheet className="h-[18px] w-[18px]" />}
          name="Customer CSV"
          subtitle="Primary join table"
          status={customerFile ? "connected" : "pending"}
          rows={customerSourceInfo ? <AnimatedNumber value={customerSourceInfo.rows} /> : "—"}
          cols={customerFile ? <AnimatedNumber value={customerColumns.length} /> : "—"}
          detail={customerFile?.name ?? "Not connected"}
        />
        <SourceOverviewCard
          icon={<Database className="h-[18px] w-[18px]" />}
          name="SQLite Database"
          subtitle={dbTables ? `${dbTables.length} table${dbTables.length === 1 ? "" : "s"} discovered` : "Loan & collateral tables"}
          status={dbFile ? "connected" : "pending"}
          rows={loanTableInfo ? <AnimatedNumber value={loanTableInfo.row_count} /> : "—"}
          cols={loanTable || collateralTable ? <AnimatedNumber value={loanColumns.length + collateralColumns.length} /> : "—"}
          detail={dbFile?.name ?? "Not connected"}
        />
        <SourceOverviewCard
          icon={<TrendingUp className="h-[18px] w-[18px]" />}
          name="Macroeconomic Data"
          subtitle="FRED macro indicators"
          status={macroColumns.length > 0 ? "connected" : "pending"}
          rows="—"
          cols={macroColumns.length > 0 ? <AnimatedNumber value={macroColumns.length} /> : "—"}
          detail={macroDateColUsed ? `Aligned on ${macroDateColUsed}` : customerFile ? "Not fetched yet" : "Upload customer data first"}
        />
      </div>

      {/* Upload / connect workspace */}
      <SourceWorkspace
        customerInputRef={customerInputRef}
        dbInputRef={dbInputRef}
        customerFile={customerFile}
        customerColumns={customerColumns}
        onCustomerFileChosen={onCustomerFileChosen}
        dbFile={dbFile}
        dbTables={dbTables}
        dbLoading={dbLoading}
        dbError={dbError}
        onDbFileChosen={onDbFileChosen}
        loanTable={loanTable}
        collateralTable={collateralTable}
        onLoanTableChange={onLoanTableChange}
        onCollateralTableChange={onCollateralTableChange}
        macroCandidates={macroCandidates}
        macroCandidatesLoading={macroCandidatesLoading}
        selectedMacroDateCol={selectedMacroDateCol}
        setSelectedMacroDateCol={setSelectedMacroDateCol}
        macroColumns={macroColumns}
        macroDateColUsed={macroDateColUsed}
        macroLoading={macroLoading}
        macroError={macroError}
        fetchMacroFeatures={fetchMacroFeatures}
        handleReFetchMacro={handleReFetchMacro}
      />

      {/* Integration flow */}
      <IntegrationFlowDiagram nodes={flowNodes} hasWarning={Boolean(integrationReport?.warnings?.length)} />

      {/* Relationships */}
      {(loanTable || collateralTable) && customerFile && dbFile ? (
        <RelationshipHealth
          loanTable={loanTable}
          collateralTable={collateralTable}
          customerColumns={customerColumns}
          loanColumns={loanColumns}
          collateralColumns={collateralColumns}
          customerLoanJoin={customerLoanJoin}
          setCustomerLoanJoin={setCustomerLoanJoin}
          loanCollateralJoin={loanCollateralJoin}
          setLoanCollateralJoin={setLoanCollateralJoin}
          customerLoanCandidate={customerLoanCandidate}
          loanCollateralCandidate={loanCollateralCandidate}
          relLoading={relLoading}
          relError={relError}
        />
      ) : null}

      {/* Integrate action */}
      <div className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white px-6 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="text-[12.5px] text-slate-500">
          {integrationReport
            ? "Sources integrated. Re-run integration any time you change a table or join selection above."
            : customerFile
            ? "Configure your data sources above, then integrate to build the modelling dataset."
            : "Upload customer data to get started."}
        </div>
        <button
          type="button"
          onClick={runIntegration}
          disabled={!canIntegrate || integrating}
          className="flex flex-shrink-0 items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-2.5 text-[13px] font-semibold text-white shadow-sm transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {integrating && <Loader className="h-4 w-4 animate-spin" />}
          {integrating ? "Integrating…" : "Integrate sources"}
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
      {integrateError ? (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{integrateError}</div>
      ) : null}

      {/* Post-integration readiness */}
      {integrationReport ? (
        <DatasetReadinessPanel
          integrationReport={integrationReport}
          joinHealthPct={joinHealthPct}
          isReady={isReady}
          downloadReport={downloadReport}
        />
      ) : null}

      {isReady && profile ? (
        <>
          <DatasetPreviewTable profile={profile} />
          <div className="flex justify-end gap-3">
            <button
              type="button"
              onClick={() => navigate({ to: "/profiling" })}
              className="flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-2.5 text-[13px] font-semibold text-white shadow-sm transition-colors hover:bg-blue-700"
            >
              Proceed to Data Profiling
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </>
      ) : (
        <div className="rounded-xl border border-slate-200 bg-white p-6">
          <div className="flex items-center gap-2">
            <Info className="h-5 w-5 text-slate-400" />
            <h4 className="text-sm font-semibold text-slate-800">Welcome to Aegis Credit</h4>
          </div>
          <p className="mt-2 text-sm text-slate-500">
            Upload customer data, connect a loan/collateral database, and optionally attach FRED macro data — the
            platform discovers how they relate and builds one integrated dataset for modelling.
          </p>
        </div>
      )}

      {/* Pipeline progress */}
      <PipelineProgress steps={pipelineSteps} />
    </div>
  );
}

// ── Presentational helpers (all props-driven — no new state/fetches) ─────────

function StatusBadge({ status, label }: { status: SourceStatus; label: string }) {
  const styles: Record<SourceStatus, { chip: string; dot: string }> = {
    connected: { chip: "bg-emerald-50 text-emerald-700", dot: "bg-emerald-500" },
    pending: { chip: "bg-slate-100 text-slate-500", dot: "bg-slate-400" },
    warning: { chip: "bg-amber-50 text-amber-700", dot: "bg-amber-500" },
  };
  const s = styles[status];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold tracking-wide transition-colors duration-300 ${s.chip}`}>
      <span className={`h-1.5 w-1.5 rounded-full transition-colors duration-300 ${s.dot}`} />
      {label}
    </span>
  );
}

// Light-on-dark badge pill for the gradient hero banner below (StatusBadge
// above is light-on-white, tuned for the white SourceOverviewCards instead).
function HeroPill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "emerald" | "amber" }) {
  const toneClasses: Record<string, string> = {
    neutral: "border-white/15 bg-white/10 text-slate-100",
    emerald: "border-emerald-300/40 bg-emerald-400/15 text-emerald-200",
    amber: "border-amber-300/40 bg-amber-400/15 text-amber-200",
  };
  return (
    <span className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-3 py-1 text-[11px] font-semibold ${toneClasses[tone]}`}>
      {children}
    </span>
  );
}

function KPITile({ label, value, sub, accent }: { label: string; value: ReactNode; sub?: string; accent?: boolean }) {
  return (
    <div className={`flex flex-col gap-1 rounded-xl border px-4 py-4 transition-colors duration-300 ${accent ? "border-blue-200 bg-blue-50" : "border-slate-200 bg-white"}`}>
      <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">{label}</span>
      <span className={`font-mono text-2xl font-semibold leading-tight transition-colors duration-300 ${accent ? "text-blue-700" : "text-slate-900"}`}>{value}</span>
      {sub && <span className="text-[11px] text-slate-400">{sub}</span>}
    </div>
  );
}

function SummaryRow({ label, value, mono }: { label: string; value: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between border-b border-slate-100 py-1.5 last:border-0">
      <span className="text-xs text-slate-500">{label}</span>
      <span className={`max-w-[60%] truncate text-right text-[12.5px] font-medium text-slate-800 ${mono ? "font-mono" : ""}`}>{value}</span>
    </div>
  );
}

function Dropzone({
  active, icon, title, subtitle, onClick, onDrop,
}: {
  active: boolean;
  icon: ReactNode;
  title: string;
  subtitle: string;
  onClick: () => void;
  onDrop: (f: File) => void;
}) {
  return (
    <div
      onClick={onClick}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        const f = e.dataTransfer?.files?.[0];
        if (f) onDrop(f);
      }}
      className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-8 text-center transition-colors ${
        active ? "border-blue-200 bg-blue-50 hover:border-blue-300" : "border-slate-200 bg-slate-50 hover:border-blue-300"
      }`}
    >
      <div className={active ? "text-blue-500" : "text-slate-400"}>{icon}</div>
      <div className={`break-all text-[13px] font-semibold ${active ? "text-blue-700" : "text-slate-600"}`}>{title}</div>
      <div className="text-[11px] text-slate-400">{subtitle}</div>
    </div>
  );
}

function ChevronSelect({
  value, onChange, children, className,
}: {
  value: string;
  onChange: (v: string) => void;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`relative ${className ?? ""}`}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full appearance-none rounded-lg border border-slate-200 bg-white px-3 py-2 text-[12.5px] text-slate-700 outline-none focus:border-blue-400"
      >
        {children}
      </select>
      <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
    </div>
  );
}

function SourceOverviewCard({
  icon, name, subtitle, status, rows, cols, detail,
}: {
  icon: ReactNode;
  name: string;
  subtitle: string;
  status: SourceStatus;
  rows: ReactNode;
  cols: ReactNode;
  detail: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 transition-shadow hover:shadow-md">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600">{icon}</div>
          <div>
            <div className="text-[13px] font-semibold text-slate-800">{name}</div>
            <div className="text-[11px] text-slate-400">{subtitle}</div>
          </div>
        </div>
        <StatusBadge status={status} label={status === "connected" ? "CONNECTED" : status === "warning" ? "WARNING" : "PENDING"} />
      </div>
      <div className="flex items-center gap-4 border-t border-slate-100 pt-3">
        <div>
          <div className="font-mono text-[15px] font-semibold text-slate-800">{rows}</div>
          <div className="text-[10px] uppercase tracking-wide text-slate-400">rows</div>
        </div>
        <div>
          <div className="font-mono text-[15px] font-semibold text-slate-800">{cols}</div>
          <div className="text-[10px] uppercase tracking-wide text-slate-400">columns</div>
        </div>
        <div className="ml-auto max-w-[45%] text-right">
          <div className="truncate font-mono text-[11px] text-slate-400" title={detail}>{detail}</div>
        </div>
      </div>
    </div>
  );
}

function SourceWorkspace({
  customerInputRef, dbInputRef,
  customerFile, customerColumns, onCustomerFileChosen,
  dbFile, dbTables, dbLoading, dbError, onDbFileChosen,
  loanTable, collateralTable, onLoanTableChange, onCollateralTableChange,
  macroCandidates, macroCandidatesLoading, selectedMacroDateCol, setSelectedMacroDateCol,
  macroColumns, macroDateColUsed, macroLoading, macroError, fetchMacroFeatures, handleReFetchMacro,
}: {
  customerInputRef: RefObject<HTMLInputElement | null>;
  dbInputRef: RefObject<HTMLInputElement | null>;
  customerFile: File | null;
  customerColumns: string[];
  onCustomerFileChosen: (f: File | null) => void;
  dbFile: File | null;
  dbTables: TableInfo[] | null;
  dbLoading: boolean;
  dbError: string | null;
  onDbFileChosen: (f: File | null) => void;
  loanTable: string;
  collateralTable: string;
  onLoanTableChange: (t: string) => void;
  onCollateralTableChange: (t: string) => void;
  macroCandidates: MacroDateCandidate[];
  macroCandidatesLoading: boolean;
  selectedMacroDateCol: string;
  setSelectedMacroDateCol: (v: string) => void;
  macroColumns: string[];
  macroDateColUsed: string | null;
  macroLoading: boolean;
  macroError: string | null;
  fetchMacroFeatures: () => void;
  handleReFetchMacro: () => void;
}) {
  const [activeTab, setActiveTab] = useState<"csv" | "sqlite" | "macro">("csv");

  const tabs: { key: "csv" | "sqlite" | "macro"; label: string; icon: ReactNode }[] = [
    { key: "csv", label: "Customer CSV", icon: <FileSpreadsheet className="h-4 w-4" /> },
    { key: "sqlite", label: "SQLite Database", icon: <Database className="h-4 w-4" /> },
    { key: "macro", label: "Macroeconomic", icon: <TrendingUp className="h-4 w-4" /> },
  ];

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
      <div className="flex border-b border-slate-200 bg-slate-50">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            type="button"
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-2 border-b-2 px-5 py-3 text-[12.5px] font-medium transition-colors ${
              activeTab === tab.key ? "border-blue-600 bg-white text-blue-700" : "border-transparent text-slate-500 hover:text-slate-700"
            }`}
          >
            <span className={activeTab === tab.key ? "text-blue-600" : "text-slate-400"}>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 p-5 md:grid-cols-2">
        {/* LEFT: controls */}
        <div>
          {activeTab === "csv" && (
            <div>
              <input
                ref={customerInputRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => onCustomerFileChosen(e.target.files?.[0] ?? null)}
              />
              <div className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-slate-400">Source File</div>
              <Dropzone
                active={Boolean(customerFile)}
                icon={<Upload className="h-6 w-6" />}
                title={customerFile ? customerFile.name : "Click or drop a CSV file here"}
                subtitle={customerFile ? "Click to replace or drag a new file" : "Supports .csv (UTF-8)."}
                onClick={() => customerInputRef.current?.click()}
                onDrop={onCustomerFileChosen}
              />
            </div>
          )}

          {activeTab === "sqlite" && (
            <div>
              <input
                ref={dbInputRef}
                type="file"
                accept=".db,.sqlite,.sqlite3"
                className="hidden"
                onChange={(e) => onDbFileChosen(e.target.files?.[0] ?? null)}
              />
              <div className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-slate-400">Database File</div>
              <Dropzone
                active={Boolean(dbFile)}
                icon={<Database className="h-6 w-6" />}
                title={dbFile ? dbFile.name : "Click or drop a SQLite file here"}
                subtitle="Accepts .db, .sqlite, .sqlite3"
                onClick={() => dbInputRef.current?.click()}
                onDrop={onDbFileChosen}
              />
              {dbLoading ? <div className="mt-2 text-xs text-slate-400">Inspecting database…</div> : null}
              {dbError ? <div className="mt-2 text-xs text-red-500">{dbError}</div> : null}

              {dbTables ? (
                <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <div>
                    <div className="mb-1.5 text-[11px] font-medium text-slate-500">Loan table</div>
                    <ChevronSelect value={loanTable} onChange={onLoanTableChange}>
                      <option value="">— none —</option>
                      {dbTables.map((t) => (
                        <option key={t.table} value={t.table}>{t.table} ({t.row_count} rows)</option>
                      ))}
                    </ChevronSelect>
                  </div>
                  <div>
                    <div className="mb-1.5 text-[11px] font-medium text-slate-500">
                      Collateral table <span className="text-slate-400">(optional)</span>
                    </div>
                    <ChevronSelect value={collateralTable} onChange={onCollateralTableChange}>
                      <option value="">— none —</option>
                      {dbTables.map((t) => (
                        <option key={t.table} value={t.table}>{t.table} ({t.row_count} rows)</option>
                      ))}
                    </ChevronSelect>
                  </div>
                </div>
              ) : null}
            </div>
          )}

          {activeTab === "macro" && (
            <div>
              {customerFile ? (
                macroColumns.length > 0 ? (
                  <div className="space-y-3">
                    <div className="rounded-xl border border-blue-200 bg-blue-50 p-4 text-sm text-blue-800">
                      Macro data attached: {macroColumns.join(", ")}
                      {macroDateColUsed ? <> (matched to the month of <code className="font-mono">{macroDateColUsed}</code>)</> : null}
                    </div>
                    <button
                      type="button"
                      onClick={handleReFetchMacro}
                      className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition-colors hover:border-blue-300 hover:bg-blue-50"
                    >
                      <RefreshCw className="h-4 w-4" />
                      Re-fetch / change date column
                    </button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-widest text-slate-400">Loan Origination Date Column</div>
                      <p className="mt-1 text-xs text-slate-500">
                        Auto-detected below — change it if a different date column is the right one to align macro data to.
                      </p>
                      {macroCandidatesLoading ? (
                        <div className="mt-2 flex w-full max-w-md items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-500">
                          <Loader className="h-4 w-4 animate-spin" />
                          Detecting date column…
                        </div>
                      ) : macroCandidates.length > 0 ? (
                        <ChevronSelect value={selectedMacroDateCol} onChange={setSelectedMacroDateCol} className="mt-2 max-w-md">
                          {macroCandidates.map(({ column, is_preferred }) => (
                            <option key={column} value={column}>
                              {is_preferred ? `⭐ ${column} (origination/loan date)` : column}
                            </option>
                          ))}
                        </ChevronSelect>
                      ) : (
                        <div className="mt-2 w-full max-w-md rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-500">
                          No date columns detected
                        </div>
                      )}
                    </div>
                    <p className="text-xs text-slate-500">
                      Optionally enrich the dataset with macroeconomic indicators from FRED. The FRED integration is configured server-side.
                    </p>
                    <button
                      type="button"
                      disabled={!selectedMacroDateCol || macroLoading || macroCandidatesLoading}
                      onClick={fetchMacroFeatures}
                      className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {macroLoading ? <Loader className="h-4 w-4 animate-spin" /> : <Globe className="h-4 w-4" />}
                      Fetch FRED macro features
                    </button>
                    {macroError ? (
                      <div className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-800">{macroError}</div>
                    ) : null}
                  </div>
                )
              ) : (
                <div className="text-sm text-slate-400">Upload a Customer CSV first to enable macroeconomic enrichment.</div>
              )}
            </div>
          )}
        </div>

        {/* RIGHT: live connection summary */}
        <div className="border-t border-slate-100 pt-6 md:border-l md:border-t-0 md:pl-6 md:pt-0">
          <div className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-slate-400">Connection Summary</div>
          {activeTab === "csv" && (
            <div>
              <SummaryRow label="File name" value={customerFile?.name ?? "—"} mono />
              <SummaryRow label="Columns detected" value={customerFile ? String(customerColumns.length) : "—"} mono />
              <SummaryRow label="Rows detected" value={customerFile ? "Available after integration" : "—"} />
              <SummaryRow
                label="Upload status"
                value={<StatusBadge status={customerFile ? "connected" : "pending"} label={customerFile ? "UPLOADED" : "PENDING"} />}
              />
            </div>
          )}
          {activeTab === "sqlite" && (
            <div>
              <SummaryRow label="Database name" value={dbFile?.name ?? "—"} mono />
              <SummaryRow label="Discovered tables" value={dbTables ? `${dbTables.length} tables` : "—"} mono />
              <SummaryRow label="Selected loan table" value={loanTable || "— none —"} mono />
              <SummaryRow label="Collateral table" value={collateralTable ? `${collateralTable} (optional)` : "— none — (optional)"} mono />
              <SummaryRow
                label="Status"
                value={<StatusBadge status={dbFile ? "connected" : "pending"} label={dbFile ? "CONNECTED" : "PENDING"} />}
              />
            </div>
          )}
          {activeTab === "macro" && (
            <div>
              <SummaryRow label="Date column" value={macroDateColUsed ?? selectedMacroDateCol ?? "—"} mono />
              <SummaryRow label="Macro availability" value={macroColumns.length > 0 ? `${macroColumns.length} series attached` : "Not fetched yet"} />
              <SummaryRow label="Alignment" value="Per-record date matching" />
              <SummaryRow
                label="Status"
                value={<StatusBadge status={macroColumns.length > 0 ? "connected" : "pending"} label={macroColumns.length > 0 ? "READY" : "PENDING"} />}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function IntegrationFlowDiagram({ nodes, hasWarning }: { nodes: FlowNode[]; hasWarning: boolean }) {
  const palette: Record<SourceStatus, { text: string; bg: string; border: string }> = {
    connected: { text: "#059669", bg: "#ECFDF5", border: "#A7F3D0" },
    warning: { text: "#D97706", bg: "#FFFBEB", border: "#FDE68A" },
    pending: { text: "#94A3B8", bg: "#F8FAFC", border: "#E2E8F0" },
  };
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-800">Data Integration Flow</h3>
          <p className="mt-0.5 text-xs text-slate-400">How source datasets become the modelling dataset</p>
        </div>
        {hasWarning ? <StatusBadge status="warning" label="JOIN WARNING" /> : null}
      </div>
      <div className="flex items-center gap-0 overflow-x-auto py-2">
        {nodes.map((node, i) => {
          const c = palette[node.state];
          return (
            <div key={node.key} className="flex items-center">
              <div
                className="flex min-w-[140px] flex-shrink-0 flex-col items-center gap-1.5 rounded-xl border px-4 py-3 transition-colors duration-500"
                style={{ background: c.bg, borderColor: c.border }}
              >
                <div className="flex items-center gap-1.5">
                  {node.state === "warning" ? (
                    <AlertTriangle className="h-3.5 w-3.5 transition-colors duration-500" style={{ color: c.text }} />
                  ) : node.state === "connected" ? (
                    <CheckCircle2 className="h-3.5 w-3.5 transition-colors duration-500" style={{ color: c.text }} />
                  ) : (
                    <span className="h-2 w-2 rounded-full bg-slate-300" />
                  )}
                  <span className="whitespace-nowrap text-xs font-semibold transition-colors duration-500" style={{ color: c.text }}>{node.label}</span>
                </div>
                <span className="text-center text-[10.5px] leading-tight text-slate-500">{node.sub}</span>
              </div>
              {i < nodes.length - 1 ? (
                <div className="flex flex-shrink-0 items-center px-1">
                  <div className="h-px w-4 bg-slate-300" />
                  <ChevronRight className="h-3.5 w-3.5 text-slate-400" />
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RelationshipCard({
  sourceLabel, targetLabel, sourceColumns, targetColumns, value, onChange, candidate,
}: {
  sourceLabel: string;
  targetLabel: string;
  sourceColumns: string[];
  targetColumns: string[];
  value: { left: string; right: string } | null;
  onChange: (v: { left: string; right: string }) => void;
  candidate: JoinCandidate | null;
}) {
  const warning = !candidate;
  const confidencePct = candidate ? Math.round(candidate.confidence * 100) : null;
  const accentColor = warning ? "#D97706" : "#059669";
  return (
    <div className={`rounded-xl border p-4 ${warning ? "border-amber-200 bg-amber-50" : "border-emerald-200 bg-emerald-50"}`}>
      <div className="mb-3 flex items-center gap-3">
        <div className="min-w-0 flex-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-center">
          <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-400">Source</div>
          <div className="truncate font-mono text-xs font-semibold text-slate-800" title={sourceLabel}>{sourceLabel}</div>
          <select
            value={value?.left ?? ""}
            onChange={(e) => onChange({ left: e.target.value, right: value?.right ?? "" })}
            className="mt-1 w-full rounded border-0 bg-transparent text-center font-mono text-[11px] text-blue-600 outline-none"
          >
            <option value="">select column</option>
            {sourceColumns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        <div className="flex flex-shrink-0 flex-col items-center gap-0.5">
          {candidate ? (
            <div className="whitespace-nowrap rounded-full px-2 py-0.5 text-[10px] font-semibold text-white transition-colors duration-300" style={{ background: accentColor }}>
              {candidate.cardinality}
            </div>
          ) : null}
          <ChevronRight className="h-4 w-4 transition-colors duration-300" style={{ color: accentColor }} />
          {confidencePct !== null ? (
            <div className="whitespace-nowrap text-[10px] text-slate-500">
              <AnimatedNumber value={confidencePct} formatter={(n) => `${Math.round(n)}% conf.`} />
            </div>
          ) : null}
        </div>

        <div className="min-w-0 flex-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-center">
          <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-400">Target</div>
          <div className="truncate font-mono text-xs font-semibold text-slate-800" title={targetLabel}>{targetLabel}</div>
          <select
            value={value?.right ?? ""}
            onChange={(e) => onChange({ left: value?.left ?? "", right: e.target.value })}
            className="mt-1 w-full rounded border-0 bg-transparent text-center font-mono text-[11px] text-blue-600 outline-none"
          >
            <option value="">select column</option>
            {targetColumns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      {confidencePct !== null ? (
        <div className="mb-2.5">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-[10px] font-medium text-slate-500">Join confidence</span>
            <span className="font-mono text-[11px] font-semibold transition-colors duration-300" style={{ color: accentColor }}>
              <AnimatedNumber value={confidencePct} formatter={(n) => `${Math.round(n)}%`} />
            </span>
          </div>
          <div className="h-1 overflow-hidden rounded-full bg-slate-200">
            <div className="h-full rounded-full transition-all duration-500 ease-out" style={{ width: `${confidencePct}%`, background: accentColor }} />
          </div>
        </div>
      ) : null}

      {candidate?.reasons?.length ? (
        <div className="space-y-1">
          {candidate.reasons.map((r, i) => (
            <div key={i} className="flex items-start gap-1.5 text-[10.5px] text-slate-500">
              <span className="mt-0.5 flex-shrink-0" style={{ color: accentColor }}>—</span>
              {r}
            </div>
          ))}
        </div>
      ) : null}

      {warning ? (
        <div className="mt-2.5 flex items-center gap-1.5 text-[10.5px] font-medium text-amber-700">
          <AlertTriangle className="h-3.5 w-3.5" />
          No suggestion — select the join columns manually
        </div>
      ) : null}
    </div>
  );
}

function RelationshipHealth({
  loanTable, collateralTable, customerColumns, loanColumns, collateralColumns,
  customerLoanJoin, setCustomerLoanJoin, loanCollateralJoin, setLoanCollateralJoin,
  customerLoanCandidate, loanCollateralCandidate, relLoading, relError,
}: {
  loanTable: string;
  collateralTable: string;
  customerColumns: string[];
  loanColumns: string[];
  collateralColumns: string[];
  customerLoanJoin: { left: string; right: string } | null;
  setCustomerLoanJoin: (v: { left: string; right: string }) => void;
  loanCollateralJoin: { left: string; right: string } | null;
  setLoanCollateralJoin: (v: { left: string; right: string }) => void;
  customerLoanCandidate: JoinCandidate | null;
  loanCollateralCandidate: JoinCandidate | null;
  relLoading: boolean;
  relError: string | null;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-800">Integration Health</h3>
          <p className="mt-0.5 text-xs text-slate-400">Detected source relationships and join quality</p>
        </div>
        {relLoading ? <span className="text-xs text-slate-400">Detecting relationships…</span> : null}
      </div>
      {relError ? <p className="mb-3 text-xs text-red-500">{relError}</p> : null}
      <div className="grid gap-4 md:grid-cols-2">
        {loanTable ? (
          <RelationshipCard
            sourceLabel="customer"
            targetLabel={loanTable}
            sourceColumns={customerColumns}
            targetColumns={loanColumns}
            value={customerLoanJoin}
            onChange={setCustomerLoanJoin}
            candidate={customerLoanCandidate}
          />
        ) : null}
        {loanTable && collateralTable ? (
          <RelationshipCard
            sourceLabel={loanTable}
            targetLabel={collateralTable}
            sourceColumns={loanColumns}
            targetColumns={collateralColumns}
            value={loanCollateralJoin}
            onChange={setLoanCollateralJoin}
            candidate={loanCollateralCandidate}
          />
        ) : null}
      </div>
    </div>
  );
}

function DatasetReadinessPanel({
  integrationReport, joinHealthPct, isReady, downloadReport,
}: {
  integrationReport: any;
  joinHealthPct: number | null;
  isReady: boolean;
  downloadReport: () => void;
}) {
  const warnings: string[] = integrationReport.warnings ?? [];
  const totalSources = (integrationReport.sources?.length ?? 0) + (integrationReport.macro_series?.length ? 1 : 0);
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-6 py-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
            <Landmark className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-800">Integrated Dataset</h3>
            <p className="text-[11px] text-slate-400">Dataset readiness summary after source integration</p>
          </div>
        </div>
        <button
          type="button"
          onClick={downloadReport}
          className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-500 transition-colors hover:text-slate-700"
        >
          <Download className="h-3.5 w-3.5" />
          Integration report
        </button>
      </div>

      <div className="p-5">
        <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          <KPITile label="Rows" value={<AnimatedNumber value={integrationReport.rows_after} />} sub="After integration" />
          <KPITile label="Columns" value={<AnimatedNumber value={integrationReport.columns_after} />} sub="Across all sources" />
          <KPITile label="Sources" value={<AnimatedNumber value={totalSources} />} sub="Connected inputs" />
          <KPITile
            label="Join Health"
            value={joinHealthPct !== null ? <AnimatedNumber value={joinHealthPct} formatter={(n) => `${Math.round(n)}%`} /> : "—"}
            sub={warnings.length ? `${warnings.length} warning${warnings.length === 1 ? "" : "s"}` : "No warnings"}
          />
          <KPITile label="Dataset Status" value={isReady ? "READY" : "PROCESSING"} sub={isReady ? "Ready for profiling" : "Finalizing"} accent />
        </div>

        {warnings.length > 0 ? (
          <div className="space-y-2">
            {warnings.map((w, i) => (
              <div key={i} className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
                <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-600" />
                <div className="text-[11.5px] text-amber-800">{w}</div>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function DatasetPreviewTable({ profile }: { profile: any }) {
  const columns: string[] = profile.columns ?? [];
  const rows: Record<string, any>[] = profile.data_preview ?? [];
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-5 py-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
            <Table2 className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-800">Dataset Preview</h3>
            <p className="text-[11px] text-slate-400">{rows.length} row{rows.length === 1 ? "" : "s"} shown · {columns.length} columns</p>
          </div>
        </div>
        <StatusBadge status="connected" label="INTEGRATED" />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[11.5px]">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50">
              <th className="whitespace-nowrap px-4 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-slate-500">#</th>
              {columns.map((c) => (
                <th key={c} className="whitespace-nowrap px-4 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-slate-500">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                <td className="whitespace-nowrap px-4 py-2 font-mono text-slate-400">{i + 1}</td>
                {columns.map((c) => (
                  <td key={`${i}-${c}`} className="whitespace-nowrap px-4 py-2 font-mono text-slate-700">{row[c] ?? ""}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PipelineProgress({ steps }: { steps: { label: string; sub: string; status: "done" | "active" | "pending" }[] }) {
  const cfg = {
    done: { dot: "bg-emerald-500 ring-emerald-100", label: "text-emerald-700", sub: "text-emerald-500", line: "bg-emerald-400" },
    active: { dot: "bg-blue-600 ring-4 ring-blue-200", label: "font-semibold text-blue-700", sub: "text-blue-500", line: "bg-slate-200" },
    pending: { dot: "bg-slate-300 ring-slate-100", label: "text-slate-400", sub: "text-slate-300", line: "bg-slate-200" },
  } as const;
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-6 py-5">
      <div className="mb-4 text-[11px] font-semibold uppercase tracking-widest text-slate-400">Integration Readiness Pipeline</div>
      <div className="flex items-center">
        {steps.map((step, i) => {
          const c = cfg[step.status];
          return (
            <div key={step.label} className="flex flex-1 items-center">
              <div className="flex flex-col items-center gap-1.5">
                <div className={`h-3 w-3 rounded-full ring-2 transition-colors duration-500 ${c.dot}`} />
                <div className={`whitespace-nowrap text-[10px] font-semibold tracking-wide transition-colors duration-500 ${c.label}`}>{step.label}</div>
                {step.sub ? <div className={`text-[9.5px] transition-colors duration-500 ${c.sub}`}>{step.sub}</div> : null}
              </div>
              {i < steps.length - 1 ? <div className={`mx-2 h-px flex-1 transition-colors duration-500 ${c.line}`} /> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

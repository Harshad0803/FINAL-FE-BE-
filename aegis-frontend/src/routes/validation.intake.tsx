import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState, type ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { api, formUpload } from "@/lib/api";
import { useDataset } from "@/lib/app-context";
import { ArrowRight, FileCheck, FileText, Upload, CheckCircle2, PlayCircle, ShieldCheck, ClipboardCheck, AlertTriangle } from "lucide-react";
import { useResumeState } from "@/hooks/use-resume-state";
import { StageHero, HeroChip, VCard, KpiStrip, VEmptyState } from "@/components/validation-ui";

export const Route = createFileRoute("/validation/intake")({
  head: () => ({ meta: [{ title: "Model Intake — Aegis Credit" }] }),
  component: Intake,
});

type IntakeDisplay = {
  title: string;
  description: string;
  modelMetadata: {
    title: string;
    description: string;
    registeredLabel: string;
    items: [string, string][];
  };
  targetDefinition: {
    title: string;
    expression: string;
    detail: string;
    baseRateLabel: string;
    baseRate: string;
    sampleSizeLabel: string;
    sampleSize: string;
  };
  riskTier: {
    title: string;
    value: string;
    description: string;
  };
  artifactTitle: string;
  artifactDescription: string;
  artifactSummary: string;
  artifacts: {
    fileName: string;
    status: string;
    timestamp: string;
    required: boolean;
  }[];
  governance: {
    title: string;
    description: string;
    status: string;
    checklist: string[];
  };
  nextStep: {
    description: string;
    label: string;
    path: string;
  };
};

type IntakeResponse = {
  demo_mode?: string;
  demo_label?: string;
  display: IntakeDisplay;
  val_intake_data?: {
    model_name?: string;
    owning_team?: string;
    model_owner?: string;
    lead_validator?: string;
    model_type?: string;
    model_version?: string;
    model_tier?: string;
    model_purpose?: string;
    mdd_text?: string;
    frameworks?: string[];
  };
  val_mdd_reported_metrics?: Record<string, any>;
  chk_inventory?: boolean;
  chk_tier?: boolean;
  chk_artifacts?: boolean;
  chk_prev_findings?: boolean;
  chk_reg_scope?: boolean;
  chk_independence?: boolean;
  chk_plan_approved?: boolean;
  chk_attestation?: boolean;
};


// Extracts just the tier number for the compact badge (e.g. "Tier 1 — High
// Risk" -> "1") — falls back to "—" rather than guessing when the real
// value doesn't contain a recognizable tier number yet.
function tierNumber(value: string): string {
  const match = value.match(/\d+/);
  return match ? match[0] : "—";
}

function Field({ label, className, children }: { label: string; className?: string; children: ReactNode }) {
  return (
    <div className={className}>
      <div className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">{label}</div>
      <div className="rounded-lg border border-slate-200 bg-slate-50/60 px-3 py-2.5">{children}</div>
    </div>
  );
}

function RequiredBadge({ required }: { required: boolean }) {
  return required ? (
    <span className="whitespace-nowrap rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700">Required</span>
  ) : (
    <span className="text-xs text-slate-400">Optional</span>
  );
}

function ArtifactStatus({ uploaded, parsed }: { uploaded: boolean; parsed?: boolean }) {
  if (parsed) {
    return <span className="whitespace-nowrap rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-700">Parsed</span>;
  }
  if (uploaded) {
    return <span className="whitespace-nowrap rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700">Uploaded</span>;
  }
  return <span className="whitespace-nowrap rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-700">Pending</span>;
}

function UploadAction({ label, accept, onFile }: { label: string; accept: string; onFile: (file: File) => void | Promise<void> }) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-1.5 text-sm font-medium text-blue-600 hover:text-blue-800">
      <Upload className="h-3.5 w-3.5" />
      {label}
      <input
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0] ?? null;
          if (f) void onFile(f);
        }}
      />
    </label>
  );
}

function ReadinessCheck({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className={`flex items-center gap-2.5 rounded-xl border px-4 py-3 ${ok ? "border-emerald-200 bg-emerald-50" : "border-amber-200 bg-amber-50"}`}>
      <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full ${ok ? "bg-emerald-500" : "bg-amber-400"}`}>
        {ok ? <CheckCircle2 className="h-3 w-3 text-white" /> : <AlertTriangle className="h-3 w-3 text-white" />}
      </span>
      <span className={`text-sm font-medium ${ok ? "text-emerald-800" : "text-amber-800"}`}>{label}</span>
    </div>
  );
}

// Neutral placeholder shown until the reviewer actually loads a demo (or
// completes a real intake) — deliberately has no plausible-looking values,
// since a prior version of this screen fetched a canned "clean" demo
// snapshot unconditionally on mount and displayed it as if it were live data.
const emptyIntake: IntakeDisplay = {
  title: "Stage 1 — Intake & Governance",
  description:
    "Capture model metadata, upload all required artifacts, and complete the governance attestation checklist before proceeding to automated validation stages.",
  modelMetadata: {
    title: "Model metadata",
    description: "Key registration details supplied by the development team.",
    registeredLabel: "Not yet registered",
    items: [],
  },
  targetDefinition: {
    title: "Target definition",
    expression: "Not yet determined",
    detail: "Load a demo or upload a dataset to populate the target definition",
    baseRateLabel: "Base rate",
    baseRate: "—",
    sampleSizeLabel: "Sample size",
    sampleSize: "—",
  },
  riskTier: {
    title: "Risk tier",
    value: "—",
    description: "Upload a dataset or load a demo to determine risk tier.",
  },
  artifactTitle: "Artifact inventory",
  artifactDescription: "Uploaded evidence to support subsequent validation stages.",
  artifactSummary: "No artifacts yet",
  artifacts: [],
  governance: {
    title: "Governance attestation",
    description: "Confirm the model and validation plan are ready to proceed.",
    status: "Not started",
    checklist: [
      "Model is registered in the model inventory",
      "Risk tier assignment has been documented",
      "Submitted artifacts cover dataset, MDD, and training code",
      "Previous validation findings (if any) have been reviewed",
      "Regulatory scope (IFRS 9 / SS1/23 / SS11/13) is identified",
      "Independent validation team has no conflict of interest",
      "Validation plan has been approved by the Head of Model Risk",
    ],
  },
  nextStep: {
    description: "Once intake is confirmed, proceed to Stage 2 data validation and automated checks.",
    label: "Proceed to Stage 2",
    path: "/validation/data-quality",
  },
};

function Intake() {
  const [intake, setIntake] = useState<IntakeDisplay>(emptyIntake);
  // True only once a demo has actually been loaded (or, in future, a real
  // intake has been saved) — gates the model-metadata/risk-tier/artifact
  // cards so they show a neutral prompt instead of pre-filled-looking data
  // before the reviewer has done anything.
  const [intakeLoaded, setIntakeLoaded] = useState(false);
  const navigate = useNavigate();
  const {
    setUploadResult,
    profile,
    setValidationIntakeData,
    setValidationMddText,
    setValidationMddMetrics,
    setValidationProfile,
    setValidationResults,
  } = useDataset();

  // Form state mirroring the Streamlit intake
  const [modelName, setModelName] = useState("");
  const [owningTeam, setOwningTeam] = useState("");
  const [modelOwner, setModelOwner] = useState("");
  const [leadValidator, setLeadValidator] = useState("");
  const [modelType, setModelType] = useState("PD (Probability of Default)");
  const [tier, setTier] = useState("Tier 2 — Medium Risk");
  const [version, setVersion] = useState("");
  const [purpose, setPurpose] = useState("");
  const [frameworks, setFrameworks] = useState<string[]>(["IFRS9", "SS1/23", "RBI"]);
  const [demoMode, setDemoMode] = useState<string | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoError, setDemoError] = useState<string | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [draftSavedAt, setDraftSavedAt] = useState<string | null>(null);
  const [proceedError, setProceedError] = useState<string | null>(null);

  // Resume where the reviewer left off: only pre-fills the (currently empty)
  // model name field from the most recently saved intake run — it doesn't
  // silently overwrite in-progress form data. The existing draft-lookup
  // effect below then surfaces its own opt-in "restore this draft?" prompt
  // once the name is filled in, same as if the reviewer had typed it.
  const { data: resumedIntake } = useResumeState<{ model_name?: string }>("validation_pipeline_log.csv", "intake");
  useEffect(() => {
    if (!modelName.trim() && resumedIntake?.model_name) {
      setModelName(resumedIntake.model_name);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumedIntake]);

  const toggleFramework = (fw: string) => {
    setFrameworks((prev) =>
      prev.includes(fw) ? prev.filter((f) => f !== fw) : [...prev, fw]
    );
  };

  // Artifacts state
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [mddFileName, setMddFileName] = useState<string | null>(null);
  const [mddText, setMddText] = useState<string | null>(null);
  const [mddMetrics, setMddMetrics] = useState<Record<string, any> | null>(null);
  const [mddDocumentPath, setMddDocumentPath] = useState<string | null>(null);
  const [trainingCodeFileName, setTrainingCodeFileName] = useState<string | null>(null);
  const [perfFileName, setPerfFileName] = useState<string | null>(null);
  const [profileFileName, setProfileFileName] = useState<string | null>(null);
  const [assumptionsFileName, setAssumptionsFileName] = useState<string | null>(null);
  const [hyperparamsFileName, setHyperparamsFileName] = useState<string | null>(null);

  // Checkboxes
  const [chkInventory, setChkInventory] = useState(false);
  const [chkTier, setChkTier] = useState(false);
  const [chkArtifacts, setChkArtifacts] = useState(false);
  const [chkPrevFindings, setChkPrevFindings] = useState(false);
  const [chkRegScope, setChkRegScope] = useState(false);
  const [chkIndependence, setChkIndependence] = useState(false);
  const [chkPlanApproved, setChkPlanApproved] = useState(false);
  const [chkAttestation, setChkAttestation] = useState(false);

  // This page's own form/checklist state lives in local useState (not
  // DatasetContext), so it needs its own small persistence layer to survive
  // a refresh — same localStorage pattern app-context.tsx uses for the
  // shared dataset/validation state. Restored once on mount; every change
  // after that re-saves the whole snapshot.
  // `isRestored` must be React state, not a ref: the write-effect below
  // needs to skip a render entirely between "restore finished" and "write
  // effect re-checks its guard" — a ref flip doesn't force that extra
  // render, so the write effect would still fire once with pre-restore
  // (default) values and clobber the just-read localStorage entry before
  // the restored state ever painted. Mirrors app-context.tsx's isHydrated.
  const [isRestored, setIsRestored] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") {
      setIsRestored(true);
      return;
    }
    try {
      const stored = window.localStorage.getItem("aegis_intake_form_state");
      if (stored) {
        const s = JSON.parse(stored);
        setIntake(s.intake ?? emptyIntake);
        setIntakeLoaded(Boolean(s.intakeLoaded));
        setModelName(s.modelName ?? "");
        setOwningTeam(s.owningTeam ?? "");
        setModelOwner(s.modelOwner ?? "");
        setLeadValidator(s.leadValidator ?? "");
        setModelType(s.modelType ?? "PD (Probability of Default)");
        setTier(s.tier ?? "Tier 2 — Medium Risk");
        setVersion(s.version ?? "");
        setPurpose(s.purpose ?? "");
        setFrameworks(Array.isArray(s.frameworks) && s.frameworks.length > 0 ? s.frameworks : ["IFRS9", "SS1/23", "RBI"]);
        setDemoMode(s.demoMode ?? null);
        setMddFileName(s.mddFileName ?? null);
        setMddText(s.mddText ?? null);
        setMddMetrics(s.mddMetrics ?? null);
        setMddDocumentPath(s.mddDocumentPath ?? null);
        setTrainingCodeFileName(s.trainingCodeFileName ?? null);
        setPerfFileName(s.perfFileName ?? null);
        setProfileFileName(s.profileFileName ?? null);
        setAssumptionsFileName(s.assumptionsFileName ?? null);
        setHyperparamsFileName(s.hyperparamsFileName ?? null);
        setChkInventory(Boolean(s.chkInventory));
        setChkTier(Boolean(s.chkTier));
        setChkArtifacts(Boolean(s.chkArtifacts));
        setChkPrevFindings(Boolean(s.chkPrevFindings));
        setChkRegScope(Boolean(s.chkRegScope));
        setChkIndependence(Boolean(s.chkIndependence));
        setChkPlanApproved(Boolean(s.chkPlanApproved));
        setChkAttestation(Boolean(s.chkAttestation));
      }
    } catch {
      // Ignore invalid stored state
    } finally {
      setIsRestored(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !isRestored) return;
    const snapshot = {
      intake, intakeLoaded, modelName, owningTeam, modelOwner, leadValidator, modelType, tier, version, purpose, frameworks,
      demoMode, mddFileName, mddText, mddMetrics, mddDocumentPath, trainingCodeFileName, perfFileName, profileFileName,
      assumptionsFileName, hyperparamsFileName, chkInventory, chkTier, chkArtifacts, chkPrevFindings, chkRegScope,
      chkIndependence, chkPlanApproved, chkAttestation,
    };
    try {
      window.localStorage.setItem("aegis_intake_form_state", JSON.stringify(snapshot));
    } catch (err) {
      console.warn("Failed to persist intake form state to localStorage:", err);
    }
  }, [
    intake, intakeLoaded, modelName, owningTeam, modelOwner, leadValidator, modelType, tier, version, purpose,
    demoMode, mddFileName, mddText, mddMetrics, trainingCodeFileName, perfFileName, profileFileName,
    assumptionsFileName, hyperparamsFileName, chkInventory, chkTier, chkArtifacts, chkPrevFindings, chkRegScope,
    chkIndependence, chkPlanApproved, chkAttestation, isRestored,
  ]);

  const datasetUploaded = Boolean(datasetFile || profile?.dataset_name);
  const mddUploaded = Boolean(mddFileName || mddText);
  const trainingCodeUploaded = Boolean(trainingCodeFileName);
  const profileUploaded = Boolean(profileFileName);
  const assumptionsUploaded = Boolean(assumptionsFileName);
  const perfUploaded = Boolean(perfFileName);
  const hyperparamsUploaded = Boolean(hyperparamsFileName);
  const readyCount = [datasetUploaded, mddUploaded, trainingCodeUploaded, profileUploaded, assumptionsUploaded, perfUploaded, hyperparamsUploaded].filter(Boolean).length;

  // Governance checklist items are intentionally NOT gated — they're an
  // honest record of what's actually true at this point in the process
  // (e.g. "no conflict of interest" may genuinely be unresolved yet), not a
  // set of boxes to force-tick. Only the attestation checkbox below — the
  // user's own confirmation that whatever state the checklist is actually
  // in is honestly represented — gates moving to the next stage.
  const handleProceed = () => {
    if (!chkAttestation) {
      setProceedError("Please confirm the information above is accurate before proceeding.");
      return;
    }
    setProceedError(null);
    setValidationIntakeData({
      model_name: modelName,
      owning_team: owningTeam,
      model_owner: modelOwner,
      lead_validator: leadValidator,
      model_type: modelType,
      model_tier: tier,
      model_version: version,
      model_purpose: purpose,
      mdd_text: mddText ?? null,
      mdd_document_path: mddDocumentPath ?? null,
      frameworks,
    });
    navigate({ to: intake.nextStep.path });
  };

  // Backend-persisted draft (keyed by model_name — no real user/session
  // system exists yet, so this is the closest thing to "this submission").
  // Deliberately opt-in rather than auto-restoring: silently overwriting
  // fields the user is mid-typing (or that a demo just populated) as soon as
  // a matching model_name is typed would be a worse surprise than just
  // asking first.
  const [draftAvailable, setDraftAvailable] = useState<{ savedAt: string; data: Record<string, any> } | null>(null);
  const [draftDismissedFor, setDraftDismissedFor] = useState<string | null>(null);
  const [draftSaving, setDraftSaving] = useState(false);
  const [draftLoadError, setDraftLoadError] = useState<string | null>(null);

  useEffect(() => {
    const trimmed = modelName.trim();
    if (!trimmed || intakeLoaded || draftDismissedFor === trimmed) {
      setDraftAvailable(null);
      return;
    }
    let active = true;
    const timer = setTimeout(() => {
      void api<{ found: boolean; saved_at?: string; data?: Record<string, any> }>(
        `/validation/intake/draft?model_name=${encodeURIComponent(trimmed)}`,
      )
        .then((resp) => {
          if (!active) return;
          if (resp.found && resp.data) {
            setDraftAvailable({ savedAt: resp.saved_at ?? "", data: resp.data });
          } else {
            setDraftAvailable(null);
          }
        })
        .catch(() => {
          if (active) setDraftAvailable(null);
        });
    }, 600);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [modelName, intakeLoaded, draftDismissedFor]);

  const applyDraft = (draft: Record<string, any>) => {
    if (draft.owningTeam) setOwningTeam(draft.owningTeam);
    if (draft.modelOwner) setModelOwner(draft.modelOwner);
    if (draft.leadValidator) setLeadValidator(draft.leadValidator);
    if (draft.modelType) setModelType(draft.modelType);
    if (draft.tier) setTier(draft.tier);
    if (draft.version) setVersion(draft.version);
    if (draft.purpose) setPurpose(draft.purpose);
    if (draft.mddFileName) setMddFileName(draft.mddFileName);
    if (draft.trainingCodeFileName) setTrainingCodeFileName(draft.trainingCodeFileName);
    if (draft.profileFileName) setProfileFileName(draft.profileFileName);
    if (draft.assumptionsFileName) setAssumptionsFileName(draft.assumptionsFileName);
    if (draft.perfFileName) setPerfFileName(draft.perfFileName);
    if (draft.hyperparamsFileName) setHyperparamsFileName(draft.hyperparamsFileName);
    setChkInventory(Boolean(draft.chkInventory));
    setChkTier(Boolean(draft.chkTier));
    setChkArtifacts(Boolean(draft.chkArtifacts));
    setChkPrevFindings(Boolean(draft.chkPrevFindings));
    setChkRegScope(Boolean(draft.chkRegScope));
    setChkIndependence(Boolean(draft.chkIndependence));
    setChkPlanApproved(Boolean(draft.chkPlanApproved));
    setChkAttestation(Boolean(draft.chkAttestation));
  };

  const loadAvailableDraft = () => {
    if (!draftAvailable) return;
    applyDraft(draftAvailable.data);
    setDraftSavedAt(draftAvailable.savedAt);
    setDraftAvailable(null);
  };

  const dismissAvailableDraft = () => {
    setDraftDismissedFor(modelName.trim());
    setDraftAvailable(null);
  };

  const saveDraft = async () => {
    const trimmed = modelName.trim();
    if (!trimmed) return;
    setDraftSaving(true);
    setDraftLoadError(null);
    try {
      const data = {
        owningTeam, modelOwner, leadValidator, modelType, tier, version, purpose,
        mddFileName, trainingCodeFileName, profileFileName, assumptionsFileName, perfFileName, hyperparamsFileName,
        chkInventory, chkTier, chkArtifacts, chkPrevFindings, chkRegScope, chkIndependence, chkPlanApproved, chkAttestation,
      };
      const resp = await api<{ saved: boolean; saved_at: string }>("/validation/intake/draft", {
        method: "POST",
        body: JSON.stringify({ model_name: trimmed, data }),
      });
      setDraftSavedAt(new Date(resp.saved_at).toLocaleString());
    } catch (err) {
      setDraftLoadError(err instanceof Error ? err.message : "Failed to save draft.");
    } finally {
      setDraftSaving(false);
    }
  };

  const demoOptions = [
    { key: "Demo A — Gold Standard", mode: "clean" },
    { key: "Demo B — Flawed Submission", mode: "flawed" },
  ];

  // Real, per-artifact source of truth for the inventory table below and the
  // readiness strip's "required uploaded" counts — required flags mirror
  // what Stage 2+ actually needs (dataset, MDD, training code), not Figma's
  // placeholder set.
  const artifactRows = [
    { key: "dataset", label: "Validation Dataset", format: "CSV / XLSX", required: true, uploaded: datasetUploaded, parsed: false, detail: "Needed for Stage 2 data checks" },
    { key: "mdd", label: "Model Development Document", format: "PDF / DOCX / TXT", required: true, uploaded: mddUploaded, parsed: Boolean(mddMetrics), detail: "Governance evidence for Stage 1" },
    { key: "training", label: "Training Code / Scripts", format: "ZIP / PY / IPYNB", required: true, uploaded: trainingCodeUploaded, parsed: false, detail: "Required for replication and review" },
    { key: "profile", label: "Data Profile", format: "CSV / XLSX / PDF", required: false, uploaded: profileUploaded, parsed: false, detail: "Helps check feature coverage" },
    { key: "assumptions", label: "Assumptions & Limitations", format: "PDF / DOCX / TXT", required: false, uploaded: assumptionsUploaded, parsed: false, detail: "Model limitations and assumptions" },
    { key: "performance", label: "Performance Report", format: "PDF / DOCX / XLSX", required: false, uploaded: perfUploaded, parsed: false, detail: "Model accuracy and stability" },
    { key: "hyperparams", label: "Hyperparameters", format: "JSON", required: false, uploaded: hyperparamsUploaded, parsed: false, detail: "Training configuration details" },
  ] as const;
  const reqTotal = artifactRows.filter((a) => a.required).length;
  const reqUploaded = artifactRows.filter((a) => a.required && a.uploaded).length;

  const attestationValues = [chkInventory, chkTier, chkArtifacts, chkPrevFindings, chkRegScope, chkIndependence, chkPlanApproved];
  const attestedCount = attestationValues.filter(Boolean).length;
  const allAttested = attestationValues.every(Boolean);

  // Metadata completeness is a real, honestly-computed signal (not a Figma
  // placeholder) — the 6 free-text fields a reviewer actually has to fill
  // in; Model Type / Risk Tier are excluded since they always carry a
  // selected default and can never be "incomplete".
  const metadataFields = [modelName, owningTeam, modelOwner, leadValidator, version, purpose];
  const metadataFilledCount = metadataFields.filter((v) => v.trim().length > 0).length;
  const metadataComplete = metadataFilledCount === metadataFields.length;

  const loadDemo = async (mode: string) => {
    setDemoError(null);
    setDemoLoading(true);
    try {
      const response = await api<IntakeResponse>(`/validation/intake?mode=${mode}`);
      if (!response.display) {
        throw new Error("Invalid demo response from backend.");
      }
      setIntake(response.display);
      setIntakeLoaded(true);
      const snapshot = response.val_intake_data;
      if (snapshot) {
        const selectedFrameworks = Array.isArray(snapshot.frameworks) && snapshot.frameworks.length > 0
          ? snapshot.frameworks
          : frameworks;
        const intakeSnapshot = {
          ...snapshot,
          mdd_text: snapshot.mdd_text ?? null,
          mdd_document_path: snapshot.mdd_document_path ?? null,
          frameworks: selectedFrameworks,
        };
        setValidationIntakeData(intakeSnapshot);
        setFrameworks(selectedFrameworks);
        setModelName(snapshot.model_name ?? "");
        setOwningTeam(snapshot.owning_team ?? "");
        setModelOwner(snapshot.model_owner ?? "");
        setLeadValidator(snapshot.lead_validator ?? "");
        setModelType(snapshot.model_type ?? "PD (Probability of Default)");
        setTier(snapshot.model_tier ?? "Tier 2 — Medium Risk");
        setVersion(snapshot.model_version ?? "");
        setPurpose(snapshot.model_purpose ?? "");
        if (snapshot.mdd_text) {
          setMddText(snapshot.mdd_text);
          setMddFileName("Parsed MDD from backend");
          setMddDocumentPath(snapshot.mdd_document_path ?? null);
          setValidationMddText(snapshot.mdd_text);
        }
      }
      setChkInventory(response.chk_inventory ?? false);
      setChkTier(response.chk_tier ?? false);
      setChkArtifacts(response.chk_artifacts ?? false);
      setChkPrevFindings(response.chk_prev_findings ?? false);
      setChkRegScope(response.chk_reg_scope ?? false);
      setChkIndependence(response.chk_independence ?? false);
      setChkPlanApproved(response.chk_plan_approved ?? false);
      setChkAttestation(response.chk_attestation ?? false);
      if (response.val_mdd_reported_metrics) {
        setMddMetrics(response.val_mdd_reported_metrics);
        setValidationMddMetrics(response.val_mdd_reported_metrics);
      }
      setDemoMode(response.demo_label ?? response.demo_mode ?? null);

      const demoForm = new FormData();
      demoForm.append("demo_mode", mode);
      const demoProfile = await formUpload("/data/upload", demoForm);
      setUploadResult(null, demoProfile as any);
      setDatasetFile(null);
      setValidationProfile(demoProfile as any);
      setValidationResults(null);
    } catch (error) {
      setDemoError(error instanceof Error ? error.message : "Unable to load demo submission.");
    } finally {
      setDemoLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <StageHero
        eyebrow="STAGE 1 · MODEL VALIDATION"
        title={intake.title}
        description={intake.description}
        chips={
          <>
            <HeroChip tone={readyCount >= 3 ? "success" : "neutral"}>{readyCount}/7 artifacts uploaded</HeroChip>
            <HeroChip tone={chkAttestation ? "success" : "neutral"}>{chkAttestation ? "Attested" : "Attestation pending"}</HeroChip>
          </>
        }
      />

      {/* Readiness strip — every value is a real, live-computed signal from
          the form/upload/attestation state below, never a fixed placeholder. */}
      <KpiStrip
        tiles={[
          {
            icon: Upload, label: "Artifacts Submitted", value: `${readyCount}/7`,
            sub: `${reqUploaded} of ${reqTotal} required uploaded`, tone: reqUploaded === reqTotal ? "emerald" : "amber",
          },
          {
            icon: ShieldCheck, label: "Governance Status", value: allAttested ? "Complete" : "Pending",
            sub: `${attestedCount} of 7 controls attested`, tone: allAttested ? "violet" : "amber",
          },
          {
            icon: FileText, label: "Model Metadata", value: metadataComplete ? "Complete" : "Incomplete",
            sub: `${metadataFilledCount} of ${metadataFields.length} fields completed`, tone: metadataComplete ? "emerald" : "amber",
          },
          {
            icon: ArrowRight, label: "Submission Readiness", value: chkAttestation ? "Ready" : "Not Ready",
            sub: chkAttestation ? "All conditions satisfied" : "Complete attestation to proceed", tone: chkAttestation ? "emerald" : "slate",
          },
        ]}
      />

      <VCard icon={PlayCircle} title="Demo mode" sub="Load a pre-configured demo intake submission instead of uploading every artifact manually.">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
          <div className="flex flex-1 flex-col gap-3 sm:flex-row">
            {demoOptions.map((demo) => {
              const [label, sub] = demo.key.split(" — ");
              return (
                <button
                  key={demo.key}
                  onClick={() => loadDemo(demo.mode)}
                  disabled={demoLoading}
                  className="flex-1 rounded-xl border border-slate-200 bg-white px-5 py-2.5 text-left transition hover:border-blue-300 hover:bg-blue-50/40 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <div className="text-sm font-semibold text-slate-700">{label}</div>
                  <div className="mt-0.5 text-xs text-slate-400">{sub}</div>
                </button>
              );
            })}
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-3">
            <Badge variant="outline" className="border-slate-200 bg-slate-50 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
              {demoMode ?? "No demo loaded"}
            </Badge>
            {demoLoading ? <span className="text-xs text-slate-500">Loading demo...</span> : null}
            {demoError ? <span className="text-xs text-amber-600">{demoError}</span> : null}
          </div>
        </div>
      </VCard>

      {/* Model profile + Risk tier / Target definition */}
      <section className="grid grid-cols-1 gap-5 xl:grid-cols-5">
        <VCard
          icon={FileText}
          title="Model Profile"
          sub={intake.modelMetadata.description}
          badge={{ text: intake.modelMetadata.registeredLabel, tone: intake.modelMetadata.registeredLabel === "Registered" ? "primary" : "slate" }}
          className="xl:col-span-3"
        >
          <div className="grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2">
            <Field label="Model Name">
              <input value={modelName} onChange={(e) => setModelName(e.target.value)} className="w-full bg-transparent font-mono text-sm font-semibold text-slate-900 outline-none" />
            </Field>
            <Field label="Model Version">
              <input value={version} onChange={(e) => setVersion(e.target.value)} className="w-full bg-transparent font-mono text-sm font-semibold text-slate-900 outline-none" />
            </Field>
            <Field label="Owning Team / Business Unit">
              <input value={owningTeam} onChange={(e) => setOwningTeam(e.target.value)} className="w-full bg-transparent text-sm font-medium text-slate-800 outline-none" />
            </Field>
            <Field label="Model Owner">
              <input value={modelOwner} onChange={(e) => setModelOwner(e.target.value)} className="w-full bg-transparent text-sm font-medium text-slate-800 outline-none" />
            </Field>
            <Field label="Lead Validator">
              <input value={leadValidator} onChange={(e) => setLeadValidator(e.target.value)} className="w-full bg-transparent text-sm font-medium text-slate-800 outline-none" />
            </Field>
            <Field label="Model Type">
              <select value={modelType} onChange={(e) => setModelType(e.target.value)} className="w-full cursor-pointer bg-transparent text-sm font-semibold text-slate-800 outline-none">
                <option>PD (Probability of Default)</option>
                <option>LGD (Loss Given Default)</option>
                <option>EAD (Exposure at Default)</option>
                <option>Scorecard / Rating</option>
              </select>
            </Field>
            <Field label="Risk Tier" className="sm:col-span-2">
              <select value={tier} onChange={(e) => setTier(e.target.value)} className="w-full cursor-pointer bg-transparent text-sm font-semibold text-blue-700 outline-none">
                <option>Tier 1 — High Risk</option>
                <option>Tier 2 — Medium Risk</option>
                <option>Tier 3 — Low Risk</option>
              </select>
            </Field>
            <div className="sm:col-span-2">
              <div className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">Model Purpose</div>
              <textarea
                value={purpose}
                onChange={(e) => setPurpose(e.target.value)}
                rows={2}
                className="w-full resize-none rounded-lg border border-slate-200 bg-slate-50/60 px-3 py-2.5 text-sm text-slate-700 focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
              />
            </div>
          </div>

          {draftAvailable ? (
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-primary/30 bg-primary-soft px-4 py-3 text-sm">
              <span>
                A saved draft exists for <strong>{modelName.trim()}</strong>
                {draftAvailable.savedAt ? ` (saved ${new Date(draftAvailable.savedAt).toLocaleString()})` : ""}.
              </span>
              <span className="flex gap-2">
                <button type="button" onClick={loadAvailableDraft} className="rounded-lg border border-primary bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground">
                  Load draft
                </button>
                <button type="button" onClick={dismissAvailableDraft} className="rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold text-foreground">
                  Dismiss
                </button>
              </span>
            </div>
          ) : null}
        </VCard>

        <div className="flex flex-col gap-4 xl:col-span-2">
          <VCard title="Risk Tier" sub="Model risk classification">
            <div className="flex items-center gap-4">
              <div className="flex h-14 w-14 shrink-0 flex-col items-center justify-center rounded-2xl bg-gradient-to-br from-blue-800 to-violet-800">
                <span className="text-[9px] font-bold uppercase tracking-widest text-white/60">Tier</span>
                <span className="text-xl font-bold leading-none text-white">{tierNumber(intake.riskTier.value)}</span>
              </div>
              <div>
                <div className="text-sm font-bold text-slate-900">{intake.riskTier.value}</div>
                <p className="mt-1 text-xs leading-snug text-slate-500">{intake.riskTier.description}</p>
              </div>
            </div>
          </VCard>

          <VCard title={intake.targetDefinition.title} className="flex-1">
            <div className="font-mono text-sm font-bold text-slate-900">{intake.targetDefinition.expression}</div>
            <p className="mt-1 text-xs text-slate-500">{intake.targetDefinition.detail}</p>
            <div className="my-4 h-px bg-slate-100" />
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">{intake.targetDefinition.baseRateLabel}</div>
                <div className="mt-1 font-mono text-sm font-bold text-slate-900">{intake.targetDefinition.baseRate}</div>
              </div>
              <div>
                <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">{intake.targetDefinition.sampleSizeLabel}</div>
                <div className="mt-1 font-mono text-sm font-bold text-slate-900">{intake.targetDefinition.sampleSize}</div>
              </div>
            </div>
          </VCard>
        </div>
      </section>

      {/* Artifact inventory — compact table; every upload/parse handler below
          is identical to the previous card-grid layout, just relocated. */}
      <VCard
        icon={FileCheck}
        title="Artifact Inventory"
        sub="Uploaded evidence to support subsequent model validation stages"
        actions={
          <div className="flex items-center gap-3">
            <span className="text-xs font-medium text-slate-500">{readyCount} / 7 uploaded</span>
            <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-100">
              <div className="h-full rounded-full bg-blue-600 transition-all" style={{ width: `${(readyCount / 7) * 100}%` }} />
            </div>
          </div>
        }
        contentClassName="-mx-6 -mb-6"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-y border-slate-100 bg-slate-50 text-left text-[10.5px] font-bold uppercase tracking-wider text-slate-400">
                <th className="px-6 py-3">Required</th>
                <th className="px-4 py-3">Artifact</th>
                <th className="px-4 py-3">Format</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-6 py-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              <tr className="hover:bg-slate-50/60">
                <td className="px-6 py-3.5"><RequiredBadge required /></td>
                <td className="px-4 py-3.5">
                  <div className="font-medium text-slate-800">Validation Dataset</div>
                  <div className="text-xs text-slate-400">{datasetFile ? datasetFile.name : profile?.dataset_name ?? "Needed for Stage 2 data checks"}</div>
                </td>
                <td className="whitespace-nowrap px-4 py-3.5 font-mono text-xs text-slate-400">CSV / XLSX</td>
                <td className="px-4 py-3.5"><ArtifactStatus uploaded={datasetUploaded} /></td>
                <td className="px-6 py-3.5 text-right">
                  <UploadAction
                    label={datasetUploaded ? "Replace" : "Attach"}
                    accept=".csv,.xlsx"
                    onFile={async (f) => {
                      setDatasetFile(f);
                      try {
                        const form = new FormData();
                        form.append("file", f);
                        const resp = await formUpload("/data/upload", form);
                        setUploadResult(f, resp as any);
                      } catch (err) {
                        console.error("Dataset upload failed", err);
                      }
                    }}
                  />
                </td>
              </tr>

              <tr className="hover:bg-slate-50/60">
                <td className="px-6 py-3.5"><RequiredBadge required /></td>
                <td className="px-4 py-3.5">
                  <div className="font-medium text-slate-800">Model Development Document</div>
                  <div className="text-xs text-slate-400">{mddFileName ?? (mddText ? `Parsed ${mddText.length} chars` : "Governance evidence for Stage 1")}</div>
                </td>
                <td className="whitespace-nowrap px-4 py-3.5 font-mono text-xs text-slate-400">PDF / DOCX / TXT</td>
                <td className="px-4 py-3.5"><ArtifactStatus uploaded={mddUploaded} parsed={Boolean(mddMetrics)} /></td>
                <td className="px-6 py-3.5 text-right">
                  <UploadAction
                    label={mddUploaded ? "Replace" : "Attach"}
                    accept=".pdf,.docx,.txt"
                    onFile={async (f) => {
                      setMddFileName(f.name);
                      setParseError(null);
                      try {
                        const form = new FormData();
                        form.append("mdd_file", f);
                        const resp = await formUpload<Record<string, any>>("/validation/parse-mdd", form);
                        setMddText(resp?.mdd_text ?? null);
                        setMddMetrics(resp?.metrics ?? null);
                        setMddDocumentPath(resp?.mdd_document_path ?? null);
                        // Publish to shared context — Stage 3's RAG keyword-search
                        // check (check_mdd_keywords) reads validationMddText from
                        // here. Without this, an MDD uploaded via this input never
                        // reaches Stage 3 and its RAG Agent Rules column stays empty.
                        setValidationMddText(resp?.mdd_text ?? null);
                        setValidationMddMetrics(resp?.metrics ?? null);
                      } catch (err) {
                        console.error("MDD parse failed", err);
                        setMddText(null);
                        setMddMetrics(null);
                        setParseError(err instanceof Error ? err.message : "Failed to parse MDD file.");
                      }
                    }}
                  />
                </td>
              </tr>

              <tr className="hover:bg-slate-50/60">
                <td className="px-6 py-3.5"><RequiredBadge required /></td>
                <td className="px-4 py-3.5">
                  <div className="font-medium text-slate-800">Training Code / Scripts</div>
                  <div className="text-xs text-slate-400">{trainingCodeFileName ?? "Required for replication and review"}</div>
                </td>
                <td className="whitespace-nowrap px-4 py-3.5 font-mono text-xs text-slate-400">ZIP / PY / IPYNB</td>
                <td className="px-4 py-3.5"><ArtifactStatus uploaded={trainingCodeUploaded} /></td>
                <td className="px-6 py-3.5 text-right">
                  <UploadAction
                    label={trainingCodeUploaded ? "Replace" : "Attach"}
                    accept=".zip,.py,.ipynb"
                    onFile={(f) => setTrainingCodeFileName(f.name)}
                  />
                </td>
              </tr>

              <tr className="hover:bg-slate-50/60">
                <td className="px-6 py-3.5"><RequiredBadge required={false} /></td>
                <td className="px-4 py-3.5">
                  <div className="font-medium text-slate-800">Data Profile</div>
                  <div className="text-xs text-slate-400">{profileFileName ?? "Helps check feature coverage"}</div>
                </td>
                <td className="whitespace-nowrap px-4 py-3.5 font-mono text-xs text-slate-400">CSV / XLSX / PDF</td>
                <td className="px-4 py-3.5"><ArtifactStatus uploaded={profileUploaded} /></td>
                <td className="px-6 py-3.5 text-right">
                  <UploadAction
                    label={profileUploaded ? "Replace" : "Attach"}
                    accept=".csv,.xlsx,.pdf"
                    onFile={(f) => setProfileFileName(f.name)}
                  />
                </td>
              </tr>

              <tr className="hover:bg-slate-50/60">
                <td className="px-6 py-3.5"><RequiredBadge required={false} /></td>
                <td className="px-4 py-3.5">
                  <div className="font-medium text-slate-800">Assumptions &amp; Limitations</div>
                  <div className="text-xs text-slate-400">{assumptionsFileName ?? "Model limitations and assumptions"}</div>
                </td>
                <td className="whitespace-nowrap px-4 py-3.5 font-mono text-xs text-slate-400">PDF / DOCX / TXT</td>
                <td className="px-4 py-3.5"><ArtifactStatus uploaded={assumptionsUploaded} /></td>
                <td className="px-6 py-3.5 text-right">
                  <UploadAction
                    label={assumptionsUploaded ? "Replace" : "Attach"}
                    accept=".pdf,.docx,.txt"
                    onFile={(f) => setAssumptionsFileName(f.name)}
                  />
                </td>
              </tr>

              <tr className="hover:bg-slate-50/60">
                <td className="px-6 py-3.5"><RequiredBadge required={false} /></td>
                <td className="px-4 py-3.5">
                  <div className="font-medium text-slate-800">Performance Report</div>
                  <div className="text-xs text-slate-400">{perfFileName ?? "Model accuracy and stability"}</div>
                </td>
                <td className="whitespace-nowrap px-4 py-3.5 font-mono text-xs text-slate-400">PDF / DOCX / XLSX</td>
                <td className="px-4 py-3.5"><ArtifactStatus uploaded={perfUploaded} /></td>
                <td className="px-6 py-3.5 text-right">
                  <UploadAction
                    label={perfUploaded ? "Replace" : "Attach"}
                    accept=".pdf,.docx,.xlsx"
                    onFile={(f) => setPerfFileName(f.name)}
                  />
                </td>
              </tr>

              <tr className="hover:bg-slate-50/60">
                <td className="px-6 py-3.5"><RequiredBadge required={false} /></td>
                <td className="px-4 py-3.5">
                  <div className="font-medium text-slate-800">Hyperparameters</div>
                  <div className="text-xs text-slate-400">{hyperparamsFileName ?? "Training configuration details"}</div>
                </td>
                <td className="whitespace-nowrap px-4 py-3.5 font-mono text-xs text-slate-400">JSON</td>
                <td className="px-4 py-3.5"><ArtifactStatus uploaded={hyperparamsUploaded} /></td>
                <td className="px-6 py-3.5 text-right">
                  <UploadAction
                    label={hyperparamsUploaded ? "Replace" : "Attach"}
                    accept=".json"
                    onFile={(f) => setHyperparamsFileName(f.name)}
                  />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        {parseError ? <div className="border-t border-slate-100 px-6 py-3 text-xs text-red-600">{parseError}</div> : null}
      </VCard>

      {/* Extracted MDD metrics — renders whatever real keys the backend
          parser actually returned; never a fixed set of metric names. */}
      <VCard
        icon={FileCheck}
        title="Extracted MDD Metrics"
        sub="Parsed from the Model Development Document"
        badge={mddMetrics ? { text: "Parsed · MDD", tone: "primary" } : undefined}
      >
        {mddMetrics ? (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {Object.entries(mddMetrics).map(([label, value]) => (
              <div key={label} className="rounded-xl border border-slate-100 bg-slate-50/60 px-4 py-3">
                <div className="text-[10px] uppercase tracking-wider text-slate-400">{label.replace(/_/g, " ")}</div>
                <div className="mt-1 font-mono text-xl font-bold text-slate-900">{value ?? "—"}</div>
              </div>
            ))}
          </div>
        ) : mddText ? (
          <div className="text-sm text-slate-500">No reported metrics were detected in the uploaded MDD.</div>
        ) : (
          <VEmptyState
            icon={FileText}
            title="No metrics extracted yet"
            description="Upload and parse a Model Development Document above to see its reported metrics here."
          />
        )}
      </VCard>

      {/* Governance */}
      <VCard icon={ShieldCheck} title="Regulatory Frameworks &amp; Governance" sub={intake.governance.description}>
        <div className="grid grid-cols-1 divide-y divide-slate-100 md:grid-cols-2 md:divide-x md:divide-y-0">
          <div className="pb-5 md:pb-0 md:pr-6">
            <div className="mb-3 text-sm font-semibold text-slate-700">Applicable Regulatory Frameworks</div>
            <div className="space-y-3">
              {[
                { key: "IFRS9", label: "IFRS 9", sub: "Financial Instruments" },
                { key: "SS1/23", label: "SS1/23", sub: "PRA Model Risk Management" },
                { key: "RBI", label: "RBI Model Risk Management", sub: "Reserve Bank of India guidance" },
              ].map((fw) => (
                <label key={fw.key} className="flex cursor-pointer items-start gap-3">
                  <input
                    type="checkbox"
                    className="mt-0.5 h-4 w-4 accent-blue-600"
                    checked={frameworks.includes(fw.key)}
                    onChange={() => toggleFramework(fw.key)}
                  />
                  <div>
                    <div className="text-sm font-medium text-slate-800">{fw.label}</div>
                    <div className="text-xs text-slate-400">{fw.sub}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className="pt-5 md:pl-6 md:pt-0">
            <div className="mb-3 text-sm font-semibold text-slate-700">{intake.governance.title}</div>
            <div className="space-y-2.5">
              {(
                [
                  [intake.governance.checklist[0], chkInventory, setChkInventory],
                  [intake.governance.checklist[1], chkTier, setChkTier],
                  [intake.governance.checklist[2], chkArtifacts, setChkArtifacts],
                  [intake.governance.checklist[3], chkPrevFindings, setChkPrevFindings],
                  [intake.governance.checklist[4], chkRegScope, setChkRegScope],
                  [intake.governance.checklist[5], chkIndependence, setChkIndependence],
                  [intake.governance.checklist[6], chkPlanApproved, setChkPlanApproved],
                ] as [string, boolean, (v: boolean) => void][]
              ).map(([label, checked, setChecked]) => (
                <label key={label} className="flex cursor-pointer items-start gap-3">
                  <input type="checkbox" className="mt-0.5 h-4 w-4 accent-blue-600" checked={checked} onChange={(e) => setChecked(e.target.checked)} />
                  <span className="text-sm leading-snug text-slate-700">{label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </VCard>

      {/* Attestation & submission — merges the previous "Attestation" card
          and the trailing "next step" card into one workspace, per Figma. */}
      <VCard icon={ClipboardCheck} title="Attestation &amp; Submission" sub={intake.nextStep.description}>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <ReadinessCheck label="Required information complete" ok={metadataComplete} />
          <ReadinessCheck label="Required evidence uploaded" ok={reqUploaded === reqTotal} />
          <ReadinessCheck label="Governance controls reviewed" ok={allAttested} />
        </div>

        <div className="my-5 border-t border-slate-100" />

        <label className="mb-5 flex cursor-pointer items-center gap-3">
          <input type="checkbox" className="h-4 w-4 accent-blue-600" checked={chkAttestation} onChange={(e) => setChkAttestation(e.target.checked)} />
          <span className="text-sm text-slate-700">I confirm the above information is accurate and complete</span>
        </label>

        {submitError ? <div className="mb-4 text-sm text-red-600">{submitError}</div> : null}
        {proceedError ? <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{proceedError}</div> : null}

        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <button
              className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
              onClick={() => void saveDraft()}
              disabled={!modelName.trim() || draftSaving}
              title={!modelName.trim() ? "Enter a model name to save a draft." : undefined}
            >
              {draftSaving ? "Saving…" : "Save draft"}
            </button>
            {!modelName.trim() ? (
              <span className="text-xs text-slate-500">Enter a model name to save a draft.</span>
            ) : draftSavedAt ? (
              <span className="text-xs text-slate-500">Draft saved {draftSavedAt} — keyed to model name "{modelName.trim()}"</span>
            ) : null}
            {draftLoadError ? <span className="text-xs text-red-600">{draftLoadError}</span> : null}
          </div>

          <div className="flex items-center gap-3">
            {!chkAttestation ? (
              <p className="flex items-center gap-1.5 text-sm text-amber-600">
                <AlertTriangle className="h-3.5 w-3.5" />
                Complete all requirements to proceed
              </p>
            ) : null}
            <button
              type="button"
              onClick={handleProceed}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-[#2f67ff] px-6 py-2.5 text-sm font-semibold text-white shadow-[0_4px_10px_rgba(47,103,255,0.18)] hover:bg-[#285ee6]"
            >
              <span>{intake.nextStep.label}</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </VCard>
    </div>
  );
}

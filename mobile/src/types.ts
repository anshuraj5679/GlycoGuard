export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN";

export type WatchPayload = {
  patient_id: string;
  glucose: number;
  roc_15?: number | null;
  trend: string;
  risk: RiskLevel;
  reason: string;
  buzz: boolean;
  forecast_30min?: number | null;
  forecast_warning?: string;
  hypo_probability?: number | null;
  top_reason?: string;
  watch_status?: string;
  updated_at: string;
  status: string;
  session_id?: string | null;
};

export const COLORS = {
  background: "#f4ecdf",
  panel: "#0b1220",
  panelSoft: "#151d2d",
  panelBorder: "rgba(255,255,255,0.06)",
  text: "#f8fafc",
  muted: "#94a3b8",
  low: "#10b981",
  medium: "#f97316",
  high: "#ef4444",
  warning: "#f59e0b",
} as const;

export function normalizeApiUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

export function buildWatchUrl(apiUrl: string, patientId: string): string {
  const baseUrl = normalizeApiUrl(apiUrl);
  if (!patientId.trim()) {
    return `${baseUrl}/watch/payload`;
  }
  return `${baseUrl}/watch/payload?patient_id=${encodeURIComponent(patientId.trim())}`;
}

export function riskSeverity(level: RiskLevel): number {
  if (level === "HIGH") {
    return 3;
  }
  if (level === "MEDIUM") {
    return 2;
  }
  if (level === "LOW") {
    return 1;
  }
  return 0;
}

export function riskColor(level: RiskLevel): string {
  if (level === "HIGH") {
    return COLORS.high;
  }
  if (level === "MEDIUM") {
    return COLORS.medium;
  }
  if (level === "LOW") {
    return COLORS.low;
  }
  return COLORS.muted;
}

export function glucoseColor(glucose: number): string {
  if (glucose < 54) {
    return COLORS.high;
  }
  if (glucose < 70) {
    return COLORS.medium;
  }
  if (glucose > 180) {
    return COLORS.warning;
  }
  return COLORS.text;
}

export function forecastColor(forecast?: number | null): string {
  if (forecast == null) {
    return COLORS.muted;
  }
  if (forecast < 70) {
    return COLORS.high;
  }
  if (forecast < 80) {
    return COLORS.medium;
  }
  return COLORS.text;
}

export function trendColor(roc15?: number | null): string {
  const value = roc15 ?? 0;
  if (value <= -2) {
    return COLORS.high;
  }
  if (value < -1) {
    return COLORS.medium;
  }
  if (value >= 2) {
    return COLORS.warning;
  }
  if (value > 1) {
    return COLORS.warning;
  }
  return COLORS.text;
}

export function trendSymbol(roc15?: number | null): string {
  const value = roc15 ?? 0;
  if (value <= -2) {
    return "\u2193\u2193";
  }
  if (value < -1) {
    return "\u2193";
  }
  if (value >= 2) {
    return "\u2191\u2191";
  }
  if (value > 1) {
    return "\u2191";
  }
  return "\u2192";
}

export function deriveDisplayRiskScore(payload: WatchPayload): number | null {
  if (payload.status !== "live" && payload.status !== "running" && payload.status !== "ok") {
    return null;
  }

  let score = Math.max(0, Math.min(1, Number(payload.hypo_probability ?? 0)));
  const forecast = payload.forecast_30min;
  const roc15 = Number(payload.roc_15 ?? 0);

  if (payload.glucose < 54) {
    score = Math.max(score, 0.99);
  }

  if (forecast != null) {
    if (forecast < 54) {
      score = Math.max(score, 0.99);
    } else if (forecast < 70) {
      score = Math.max(score, 0.85);
    } else if (forecast < 80) {
      score = Math.max(score, 0.6);
    } else if (forecast < 90 && roc15 <= -2) {
      score = Math.max(score, 0.45);
    } else if (forecast < 90 && roc15 <= -1) {
      score = Math.max(score, 0.35);
    }
  }

  if (payload.risk === "HIGH") {
    score = Math.max(score, 0.8);
  } else if (payload.risk === "MEDIUM") {
    score = Math.max(score, 0.5);
  }

  return Math.max(0, Math.min(1, score));
}

export function formatRiskScore(payload: WatchPayload): string {
  const score = deriveDisplayRiskScore(payload);
  return score == null ? "unknown" : `${Math.round(score * 100)}%`;
}

export function formatUpdatedAt(updatedAt?: string): string {
  if (!updatedAt) {
    return "--";
  }
  const date = new Date(updatedAt);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

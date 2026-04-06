import React from "react";
import { StyleSheet, Text, View } from "react-native";

import {
  COLORS,
  WatchPayload,
  deriveDisplayRiskScore,
  forecastColor,
  formatRiskScore,
  glucoseColor,
  riskColor,
  trendColor,
  trendSymbol,
} from "./types";

type Props = {
  payload: WatchPayload | null;
};

function fallbackPayload(): WatchPayload {
  return {
    patient_id: "waiting",
    glucose: 0,
    trend: "Waiting for local GlycoGuard API",
    risk: "UNKNOWN",
    reason: "Connect your phone to the same Wi-Fi network and enter your computer LAN IP.",
    buzz: false,
    forecast_30min: null,
    forecast_warning: "",
    hypo_probability: null,
    top_reason: "No live data yet.",
    watch_status: "Prediction unavailable",
    updated_at: new Date().toISOString(),
    status: "setup_required",
  };
}

export default function RiskCard({ payload }: Props) {
  const active = payload ?? fallbackPayload();
  const badgeColor = riskColor(active.risk);
  const score = deriveDisplayRiskScore(active);
  const scorePercent = score == null ? 0 : Math.round(score * 100);
  const forecastText =
    active.forecast_30min == null ? "Forecast unavailable" : `30 min: ${Math.round(active.forecast_30min)} mg/dL`;
  const reason = active.top_reason?.trim() || active.reason || "Check dashboard";
  const detailText =
    active.forecast_warning?.trim() ||
    (active.forecast_30min == null ? "Forecast unavailable while the API abstains." : "Trajectory remains above the danger buffer.");

  return (
    <View style={styles.card}>
      <View style={[styles.badge, { backgroundColor: badgeColor }]}>
        <Text style={styles.badgeText}>{active.risk} RISK</Text>
      </View>

      <View style={styles.heroRow}>
        <View>
          <Text style={[styles.glucoseValue, { color: glucoseColor(active.glucose) }]}>
            {active.glucose ? Math.round(active.glucose) : "--"}
          </Text>
          <Text style={styles.glucoseLabel}>mg/dL current glucose</Text>
        </View>
        <Text style={[styles.trendSymbol, { color: trendColor(active.roc_15) }]}>{trendSymbol(active.roc_15)}</Text>
      </View>

      <Text style={[styles.trendText, { color: trendColor(active.roc_15) }]}>{active.trend}</Text>

      <View style={styles.segmentRow}>
        {(["LOW", "MEDIUM", "HIGH"] as const).map((level) => (
          <View
            key={level}
            style={[
              styles.segment,
              active.risk === level ? { backgroundColor: badgeColor } : styles.segmentIdle,
            ]}
          >
            <Text style={active.risk === level ? styles.segmentTextActive : styles.segmentTextIdle}>{level}</Text>
          </View>
        ))}
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionLabel}>30-minute forecast</Text>
        <Text style={[styles.sectionValue, { color: forecastColor(active.forecast_30min) }]}>{forecastText}</Text>
        <Text style={[styles.sectionHint, { color: forecastColor(active.forecast_30min) }]}>
          {active.forecast_warning ? "\u26A0 " : ""}
          {detailText}
        </Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionLabel}>Top reason</Text>
        <Text style={styles.reasonText}>{reason}</Text>
      </View>

      <View style={styles.scoreBlock}>
        <View style={styles.scoreRow}>
          <Text style={styles.scoreLabel}>Risk score</Text>
          <Text style={styles.scoreValue}>{formatRiskScore(active)}</Text>
        </View>
        <View style={styles.scoreTrack}>
          <View style={[styles.scoreFill, { width: `${scorePercent}%`, backgroundColor: badgeColor }]} />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={[styles.watchStatus, { color: badgeColor }]}>
          {active.buzz ? "\u26A1 " : active.risk === "MEDIUM" ? "\u26A0 " : active.risk === "LOW" ? "\u2713 " : "\u25CB "}
          {active.watch_status || "Prediction unavailable"}
        </Text>
        <Text style={styles.watchHint}>
          {active.buzz ? "Local Android alert fired from the app." : "No watch buzz triggered right now."}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: COLORS.panel,
    borderRadius: 30,
    padding: 22,
    gap: 18,
    shadowColor: "#000000",
    shadowOpacity: 0.24,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 10 },
    elevation: 8,
  },
  badge: {
    alignSelf: "flex-start",
    borderRadius: 999,
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  badgeText: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "800",
    letterSpacing: 0.8,
  },
  heroRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-end",
  },
  glucoseValue: {
    fontSize: 62,
    lineHeight: 62,
    fontWeight: "900",
    letterSpacing: -2,
  },
  glucoseLabel: {
    color: COLORS.muted,
    fontSize: 14,
    marginTop: 8,
  },
  trendSymbol: {
    fontSize: 44,
    lineHeight: 44,
    fontWeight: "800",
  },
  trendText: {
    fontSize: 15,
    fontWeight: "700",
  },
  segmentRow: {
    flexDirection: "row",
    gap: 10,
  },
  segment: {
    flex: 1,
    borderRadius: 999,
    paddingVertical: 12,
    alignItems: "center",
  },
  segmentIdle: {
    backgroundColor: "#21283a",
  },
  segmentTextActive: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: "800",
  },
  segmentTextIdle: {
    color: COLORS.muted,
    fontSize: 13,
    fontWeight: "800",
  },
  section: {
    backgroundColor: COLORS.panelSoft,
    borderRadius: 22,
    padding: 18,
    borderWidth: 1,
    borderColor: COLORS.panelBorder,
    gap: 10,
  },
  sectionLabel: {
    color: COLORS.muted,
    textTransform: "uppercase",
    fontSize: 12,
    letterSpacing: 1.3,
    fontWeight: "700",
  },
  sectionValue: {
    fontSize: 20,
    fontWeight: "800",
  },
  sectionHint: {
    fontSize: 13,
    lineHeight: 20,
  },
  reasonText: {
    color: COLORS.text,
    fontSize: 16,
    lineHeight: 22,
    fontWeight: "600",
  },
  scoreBlock: {
    gap: 10,
  },
  scoreRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  scoreLabel: {
    color: COLORS.muted,
    fontSize: 15,
  },
  scoreValue: {
    color: COLORS.muted,
    fontSize: 15,
    fontWeight: "700",
  },
  scoreTrack: {
    height: 10,
    borderRadius: 999,
    overflow: "hidden",
    backgroundColor: "#2a3245",
  },
  scoreFill: {
    height: "100%",
    minWidth: 10,
    borderRadius: 999,
  },
  watchStatus: {
    fontSize: 16,
    lineHeight: 22,
    fontWeight: "800",
  },
  watchHint: {
    color: COLORS.muted,
    fontSize: 13,
    lineHeight: 19,
  },
});

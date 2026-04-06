import React from "react";
import { StyleSheet, Text, View } from "react-native";

import { COLORS, PatientReport, forecastColor, formatRiskScore, glucoseColor, riskColor, trendColor, trendSymbol } from "./types";

type Props = {
  report: PatientReport | null;
};

function percent(value?: number): string {
  if (value == null) {
    return "--";
  }
  return `${Math.round(value * 100)}%`;
}

function reading(value?: number | null, suffix = ""): string {
  if (value == null) {
    return "--";
  }
  return `${Number(value).toFixed(1)}${suffix}`;
}

export default function ReportCard({ report }: Props) {
  if (!report) {
    return (
      <View style={styles.card}>
        <Text style={styles.title}>Patient report</Text>
        <Text style={styles.emptyText}>Connect to GlycoGuard to load the full patient report and enable PDF download.</Text>
      </View>
    );
  }

  const prediction = report.prediction;
  const agpSummary = report.agp?.summary;
  const profile = report.profile;
  const activeRisk = prediction.risk_level ?? "UNKNOWN";
  const badgeColor = riskColor(activeRisk);
  const factorLines = (prediction.top_factors ?? []).slice(0, 3);
  const alertLines = report.alert_log.slice(-3).reverse();

  return (
    <View style={styles.card}>
      <View style={styles.headerRow}>
        <View>
          <Text style={styles.title}>Patient report</Text>
          <Text style={styles.subtitle}>{profile.name?.trim() || report.patient_id}</Text>
        </View>
        <View style={[styles.badge, { backgroundColor: badgeColor }]}>
          <Text style={styles.badgeText}>{activeRisk} RISK</Text>
        </View>
      </View>

      <View style={styles.gridRow}>
        <View style={styles.metricPanel}>
          <Text style={styles.metricLabel}>Current glucose</Text>
          <Text style={[styles.metricValue, { color: glucoseColor(report.current_glucose) }]}>{reading(report.current_glucose, " mg/dL")}</Text>
        </View>
        <View style={styles.metricPanel}>
          <Text style={styles.metricLabel}>30-min forecast</Text>
          <Text style={[styles.metricValue, { color: forecastColor(prediction.predicted_glucose_30min) }]}>
            {reading(prediction.predicted_glucose_30min, " mg/dL")}
          </Text>
        </View>
      </View>

      <View style={styles.gridRow}>
        <View style={styles.metricPanel}>
          <Text style={styles.metricLabel}>Trend</Text>
          <Text style={[styles.metricValue, { color: trendColor(report.roc_15) }]}>
            {trendSymbol(report.roc_15)} {reading(report.roc_15, " mg/dL")}
          </Text>
        </View>
        <View style={styles.metricPanel}>
          <Text style={styles.metricLabel}>Risk score</Text>
          <Text style={styles.metricValue}>
            {prediction.hypo_probability == null ? formatRiskScore({
              patient_id: report.patient_id,
              glucose: report.current_glucose,
              roc_15: report.roc_15,
              trend: "",
              risk: activeRisk,
              reason: "",
              buzz: false,
              forecast_30min: prediction.predicted_glucose_30min,
              hypo_probability: prediction.hypo_probability,
              updated_at: "",
              status: prediction.status,
            }) : percent(prediction.hypo_probability)}
          </Text>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionLabel}>Summary</Text>
        <Text style={styles.sectionValue}>{prediction.watch_status?.trim() || "Prediction unavailable"}</Text>
        <Text style={styles.sectionHint}>{prediction.top_reason?.trim() || prediction.explanation?.trim() || "No explanation available."}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionLabel}>Profile</Text>
        <Text style={styles.sectionHint}>Diabetes: {profile.diabetes_type || "N/A"}</Text>
        <Text style={styles.sectionHint}>Therapy: {profile.insulin_therapy || "N/A"}</Text>
        <Text style={styles.sectionHint}>Age: {profile.age ?? "N/A"}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionLabel}>Recent context</Text>
        <Text style={styles.sectionHint}>Carbs 1h: {reading(report.context.carbs_1h, " g")}</Text>
        <Text style={styles.sectionHint}>Insulin on board: {reading(report.context.insulin_on_board, " U")}</Text>
        <Text style={styles.sectionHint}>Activity: {reading(report.context.activity)}</Text>
        <Text style={styles.sectionHint}>Stress: {reading(report.context.stress_score)}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionLabel}>Top model factors</Text>
        {factorLines.length ? (
          factorLines.map((factor) => (
            <Text key={`${factor.feature}-${factor.message}`} style={styles.sectionHint}>
              - {factor.message}
            </Text>
          ))
        ) : (
          <Text style={styles.sectionHint}>No factor breakdown available.</Text>
        )}
      </View>

      <View style={styles.gridRow}>
        <View style={styles.metricPanel}>
          <Text style={styles.metricLabel}>Time in range</Text>
          <Text style={styles.metricValue}>{percent(agpSummary?.time_in_range)}</Text>
        </View>
        <View style={styles.metricPanel}>
          <Text style={styles.metricLabel}>Mean glucose</Text>
          <Text style={styles.metricValue}>{reading(agpSummary?.mean_glucose, " mg/dL")}</Text>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionLabel}>Recent alerts</Text>
        {alertLines.length ? (
          alertLines.map((entry) => (
            <Text key={entry.timestamp} style={styles.sectionHint}>
              - {entry.risk_level} risk, score {percent(entry.hypo_probability)}{entry.actual_hypo ? ", actual hypo" : ""}
            </Text>
          ))
        ) : (
          <Text style={styles.sectionHint}>No recent alerts recorded.</Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#fffaf3",
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    borderColor: "rgba(23,32,45,0.08)",
    gap: 12,
  },
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 12,
  },
  title: {
    color: "#17202d",
    fontSize: 20,
    fontWeight: "800",
  },
  subtitle: {
    color: "#6b7280",
    fontSize: 14,
    marginTop: 4,
  },
  badge: {
    borderRadius: 999,
    paddingHorizontal: 14,
    paddingVertical: 9,
  },
  badgeText: {
    color: COLORS.text,
    fontSize: 13,
    fontWeight: "800",
  },
  gridRow: {
    flexDirection: "row",
    gap: 10,
  },
  metricPanel: {
    flex: 1,
    backgroundColor: "#ffffff",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "rgba(23,32,45,0.08)",
    gap: 6,
  },
  metricLabel: {
    color: "#6b7280",
    fontSize: 12,
    textTransform: "uppercase",
    letterSpacing: 0.8,
    fontWeight: "700",
  },
  metricValue: {
    color: "#17202d",
    fontSize: 19,
    fontWeight: "800",
  },
  section: {
    backgroundColor: "#ffffff",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "rgba(23,32,45,0.08)",
    gap: 6,
  },
  sectionLabel: {
    color: "#6b7280",
    fontSize: 12,
    textTransform: "uppercase",
    letterSpacing: 0.8,
    fontWeight: "700",
  },
  sectionValue: {
    color: "#17202d",
    fontSize: 16,
    fontWeight: "800",
  },
  sectionHint: {
    color: "#4b5563",
    fontSize: 14,
    lineHeight: 20,
  },
  emptyText: {
    color: "#5f6b7a",
    fontSize: 14,
    lineHeight: 20,
  },
});

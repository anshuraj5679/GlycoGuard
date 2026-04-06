import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  AppState,
  AppStateStatus,
  Linking,
  Pressable,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import * as Notifications from "expo-notifications";

import ReportCard from "./src/ReportCard";
import RiskCard from "./src/RiskCard";
import {
  COLORS,
  PatientReport,
  WatchPayload,
  buildReportPdfUrl,
  buildReportUrl,
  buildWatchUrl,
  formatUpdatedAt,
  normalizeApiUrl,
  riskSeverity,
} from "./src/types";

const DEFAULT_API_URL = (process.env.EXPO_PUBLIC_GLYCOGUARD_API_URL ?? "").trim();
const DEFAULT_PATIENT_ID = (process.env.EXPO_PUBLIC_GLYCOGUARD_PATIENT_ID ?? "").trim();
const POLL_INTERVAL_MS = 5000;

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
    shouldShowBanner: true,
    shouldShowList: true,
  }),
});

async function ensureAndroidChannel(): Promise<void> {
  await Notifications.setNotificationChannelAsync("glycoguard-alerts", {
    name: "GlycoGuard Alerts",
    importance: Notifications.AndroidImportance.MAX,
    vibrationPattern: [0, 250, 150, 250],
    lightColor: COLORS.high,
    sound: "default",
  });
}

async function requestNotificationAccess(): Promise<boolean> {
  await ensureAndroidChannel();
  const current = await Notifications.getPermissionsAsync();
  if (current.granted) {
    return true;
  }
  const requested = await Notifications.requestPermissionsAsync();
  return requested.granted;
}

async function fetchWatchPayload(apiUrl: string, patientId: string): Promise<WatchPayload> {
  const response = await fetch(buildWatchUrl(apiUrl, patientId), {
    headers: { Accept: "application/json" },
  });
  const body = (await response.json()) as WatchPayload | { detail?: string };
  if (!response.ok) {
    throw new Error(typeof body === "object" && body && "detail" in body && body.detail ? String(body.detail) : "Unable to fetch watch payload.");
  }
  return body as WatchPayload;
}

async function fetchPatientReport(apiUrl: string, patientId: string): Promise<PatientReport> {
  const response = await fetch(buildReportUrl(apiUrl, patientId), {
    headers: { Accept: "application/json" },
  });
  const body = (await response.json()) as PatientReport | { detail?: string };
  if (!response.ok) {
    throw new Error(typeof body === "object" && body && "detail" in body && body.detail ? String(body.detail) : "Unable to fetch patient report.");
  }
  return body as PatientReport;
}

function shouldNotify(previous: WatchPayload | null, next: WatchPayload): boolean {
  if (next.status !== "live" && next.status !== "running" && next.status !== "ok") {
    return false;
  }
  if (!previous) {
    return false;
  }
  const previousSeverity = riskSeverity(previous.risk);
  const nextSeverity = riskSeverity(next.risk);
  if (next.buzz && !previous.buzz) {
    return true;
  }
  if (nextSeverity > previousSeverity) {
    return true;
  }
  if (next.risk === "HIGH" && previous.forecast_30min != null && next.forecast_30min != null && next.forecast_30min < previous.forecast_30min - 5) {
    return true;
  }
  return false;
}

async function fireLocalAlert(payload: WatchPayload): Promise<void> {
  const title = payload.buzz ? "GlycoGuard urgent alert" : `GlycoGuard ${payload.risk.toLowerCase()} risk`;
  const body =
    payload.watch_status?.trim() ||
    payload.forecast_warning?.trim() ||
    payload.top_reason?.trim() ||
    payload.reason ||
    "Glucose trajectory changed.";

  await Notifications.scheduleNotificationAsync({
    content: {
      title,
      body,
      sound: "default",
      priority: Notifications.AndroidNotificationPriority.MAX,
      data: {
        risk: payload.risk,
        forecast30m: payload.forecast_30min ?? null,
        glucose: payload.glucose,
      },
    },
    trigger: null,
  });
}

export default function App() {
  const [apiUrlDraft, setApiUrlDraft] = useState(DEFAULT_API_URL);
  const [patientIdDraft, setPatientIdDraft] = useState(DEFAULT_PATIENT_ID);
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [patientId, setPatientId] = useState(DEFAULT_PATIENT_ID);
  const [payload, setPayload] = useState<WatchPayload | null>(null);
  const [report, setReport] = useState<PatientReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(DEFAULT_API_URL ? null : "Enter your computer's LAN API URL to connect.");
  const [reportNote, setReportNote] = useState<string | null>(null);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);

  const lastPayloadRef = useRef<WatchPayload | null>(null);

  const refreshPayload = useCallback(async (showSpinner: boolean) => {
    const normalizedApiUrl = normalizeApiUrl(apiUrl);
    if (!normalizedApiUrl) {
      setError("Enter your computer's LAN API URL to connect.");
      return;
    }

    if (showSpinner) {
      setLoading(true);
    }

    try {
      const [payloadResult, reportResult] = await Promise.allSettled([
        fetchWatchPayload(normalizedApiUrl, patientId),
        fetchPatientReport(normalizedApiUrl, patientId),
      ]);

      if (payloadResult.status === "fulfilled") {
        const nextPayload = payloadResult.value;
        setPayload(nextPayload);
        setError(null);
        if (notificationsEnabled && shouldNotify(lastPayloadRef.current, nextPayload)) {
          await fireLocalAlert(nextPayload);
        }
        lastPayloadRef.current = nextPayload;
      } else {
        throw payloadResult.reason;
      }

      if (reportResult.status === "fulfilled") {
        setReport(reportResult.value);
        setReportNote(null);
      } else {
        setReport(null);
        setReportNote(reportResult.reason instanceof Error ? reportResult.reason.message : "Patient report is unavailable right now.");
      }
    } catch (fetchError) {
      setReport(null);
      setError(fetchError instanceof Error ? fetchError.message : "Unable to reach GlycoGuard.");
    } finally {
      if (showSpinner) {
        setLoading(false);
      }
    }
  }, [apiUrl, notificationsEnabled, patientId]);

  const downloadPdf = useCallback(async () => {
    const normalizedApiUrl = normalizeApiUrl(apiUrl);
    if (!normalizedApiUrl) {
      setError("Enter your computer's LAN API URL to connect.");
      return;
    }
    try {
      const separator = buildReportPdfUrl(normalizedApiUrl, patientId).includes("?") ? "&" : "?";
      await Linking.openURL(`${buildReportPdfUrl(normalizedApiUrl, patientId)}${separator}ts=${Date.now()}`);
      setReportNote("Opened the PDF download in your phone browser.");
    } catch (downloadError) {
      setReportNote(downloadError instanceof Error ? downloadError.message : "Unable to open the PDF download.");
    }
  }, [apiUrl, patientId]);

  const enableNotifications = useCallback(async () => {
    const granted = await requestNotificationAccess();
    setNotificationsEnabled(granted);
    if (!granted) {
      setError("Android notifications are disabled. Enable them in system settings to get alerts.");
    } else {
      setError(null);
    }
  }, []);

  useEffect(() => {
    void ensureAndroidChannel();
    void requestNotificationAccess().then(setNotificationsEnabled);

    const subscription = AppState.addEventListener("change", (state: AppStateStatus) => {
      if (state === "active" && apiUrl) {
        void refreshPayload(false);
      }
    });

    return () => {
      subscription.remove();
    };
  }, [apiUrl, refreshPayload]);

  useEffect(() => {
    if (!apiUrl) {
      return;
    }
    void refreshPayload(true);
    const timer = setInterval(() => {
      void refreshPayload(false);
    }, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [apiUrl, patientId, refreshPayload]);

  const applyConnection = () => {
    const normalizedApiUrl = normalizeApiUrl(apiUrlDraft);
    setApiUrl(normalizedApiUrl);
    setPatientId(patientIdDraft.trim());
    setPayload(null);
    setReport(null);
    setReportNote(null);
    lastPayloadRef.current = null;
    if (!normalizedApiUrl) {
      setError("Enter your computer's LAN API URL to connect.");
      return;
    }
    void refreshPayload(true);
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" />
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.hero}>
          <Text style={styles.kicker}>ANDROID CLIENT</Text>
          <Text style={styles.title}>GlycoGuard Mobile</Text>
          <Text style={styles.subtitle}>
            This app mirrors the live risk card from your local GlycoGuard API and raises Android alerts when risk escalates.
          </Text>
        </View>

        <View style={styles.settingsCard}>
          <Text style={styles.settingsTitle}>Local connection</Text>
          <Text style={styles.settingsHint}>Use your computer LAN IP, not localhost. Example: http://192.168.1.20:8000</Text>

          <Text style={styles.label}>API URL</Text>
          <TextInput
            autoCapitalize="none"
            autoCorrect={false}
            placeholder="http://192.168.1.20:8000"
            placeholderTextColor="#9ca3af"
            style={styles.input}
            value={apiUrlDraft}
            onChangeText={setApiUrlDraft}
          />

          <Text style={styles.label}>Patient ID (optional)</Text>
          <TextInput
            autoCapitalize="none"
            autoCorrect={false}
            placeholder="Leave blank for default patient"
            placeholderTextColor="#9ca3af"
            style={styles.input}
            value={patientIdDraft}
            onChangeText={setPatientIdDraft}
          />

          <View style={styles.buttonRow}>
            <Pressable style={styles.primaryButton} onPress={applyConnection}>
              <Text style={styles.primaryButtonText}>{loading ? "Connecting..." : "Connect"}</Text>
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={() => void refreshPayload(true)}>
              <Text style={styles.secondaryButtonText}>Refresh</Text>
            </Pressable>
          </View>

          <View style={styles.metaRow}>
            <Text style={styles.metaText}>Last update: {formatUpdatedAt(payload?.updated_at)}</Text>
            <Pressable onPress={() => void enableNotifications()}>
              <Text style={styles.metaLink}>{notificationsEnabled ? "Alerts enabled" : "Enable alerts"}</Text>
            </Pressable>
          </View>

          <Pressable onPress={() => Linking.openSettings()}>
            <Text style={styles.metaLink}>Open Android app settings</Text>
          </Pressable>

          <View style={styles.buttonRow}>
            <Pressable style={styles.primaryButton} onPress={() => void downloadPdf()}>
              <Text style={styles.primaryButtonText}>Download PDF report</Text>
            </Pressable>
          </View>
        </View>

        {loading && !payload ? (
          <View style={styles.loadingCard}>
            <ActivityIndicator size="large" color={COLORS.medium} />
            <Text style={styles.loadingText}>Fetching live watch payload...</Text>
          </View>
        ) : (
          <RiskCard payload={payload} />
        )}

        {error ? (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>Connection note</Text>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        ) : null}

        {reportNote ? (
          <View style={styles.infoCard}>
            <Text style={styles.infoTitle}>Report note</Text>
            <Text style={styles.infoText}>{reportNote}</Text>
          </View>
        ) : null}

        <ReportCard report={report} />

        <View style={styles.footerCard}>
          <Text style={styles.footerTitle}>What works now</Text>
          <Text style={styles.footerItem}>- Live polling from your local GlycoGuard API over Wi-Fi</Text>
          <Text style={styles.footerItem}>- Local Android notifications when MEDIUM/HIGH risk escalates</Text>
          <Text style={styles.footerItem}>- Same risk-card styling as the dashboard watch card</Text>
          <Text style={styles.footerItem}>- Full patient report fetch with PDF download through the mobile app</Text>
          <Text style={styles.footerTitle}>Practical limit</Text>
          <Text style={styles.footerItem}>
            - Near-real-time alerts while the app is open. Reliable background alerts will need FCM or a native foreground service.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  container: {
    padding: 18,
    gap: 18,
  },
  hero: {
    paddingTop: 8,
    gap: 8,
  },
  kicker: {
    color: "#7c6f5c",
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 1.4,
  },
  title: {
    color: "#17202d",
    fontSize: 34,
    fontWeight: "900",
    letterSpacing: -1,
  },
  subtitle: {
    color: "#5f6b7a",
    fontSize: 16,
    lineHeight: 23,
    maxWidth: 520,
  },
  settingsCard: {
    backgroundColor: "#fffaf3",
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    borderColor: "rgba(23,32,45,0.08)",
    gap: 10,
  },
  settingsTitle: {
    color: "#17202d",
    fontSize: 18,
    fontWeight: "800",
  },
  settingsHint: {
    color: "#6b7280",
    fontSize: 14,
    lineHeight: 20,
  },
  label: {
    color: "#3a4554",
    fontSize: 13,
    fontWeight: "700",
    marginTop: 4,
  },
  input: {
    backgroundColor: "#ffffff",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(23,32,45,0.12)",
    paddingHorizontal: 14,
    paddingVertical: 12,
    color: "#17202d",
    fontSize: 15,
  },
  buttonRow: {
    flexDirection: "row",
    gap: 10,
    marginTop: 4,
  },
  primaryButton: {
    flex: 1,
    backgroundColor: COLORS.panel,
    borderRadius: 16,
    paddingVertical: 14,
    alignItems: "center",
  },
  primaryButtonText: {
    color: COLORS.text,
    fontSize: 15,
    fontWeight: "800",
  },
  secondaryButton: {
    minWidth: 110,
    backgroundColor: "#f0e4d2",
    borderRadius: 16,
    paddingVertical: 14,
    paddingHorizontal: 18,
    alignItems: "center",
  },
  secondaryButtonText: {
    color: "#17202d",
    fontSize: 15,
    fontWeight: "800",
  },
  metaRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 4,
  },
  metaText: {
    color: "#6b7280",
    fontSize: 13,
  },
  metaLink: {
    color: "#a65412",
    fontSize: 13,
    fontWeight: "700",
  },
  loadingCard: {
    backgroundColor: COLORS.panel,
    borderRadius: 30,
    padding: 28,
    alignItems: "center",
    gap: 14,
  },
  loadingText: {
    color: COLORS.text,
    fontSize: 15,
  },
  errorCard: {
    backgroundColor: "#fff4ef",
    borderRadius: 20,
    padding: 16,
    borderWidth: 1,
    borderColor: "rgba(239,68,68,0.16)",
    gap: 6,
  },
  errorTitle: {
    color: "#991b1b",
    fontSize: 14,
    fontWeight: "800",
  },
  errorText: {
    color: "#7f1d1d",
    fontSize: 14,
    lineHeight: 20,
  },
  infoCard: {
    backgroundColor: "#fffaf3",
    borderRadius: 20,
    padding: 16,
    borderWidth: 1,
    borderColor: "rgba(23,32,45,0.08)",
    gap: 6,
  },
  infoTitle: {
    color: "#17202d",
    fontSize: 14,
    fontWeight: "800",
  },
  infoText: {
    color: "#5f6b7a",
    fontSize: 14,
    lineHeight: 20,
  },
  footerCard: {
    backgroundColor: "#fffaf3",
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    borderColor: "rgba(23,32,45,0.08)",
    gap: 8,
  },
  footerTitle: {
    color: "#17202d",
    fontSize: 16,
    fontWeight: "800",
    marginTop: 4,
  },
  footerItem: {
    color: "#5f6b7a",
    fontSize: 14,
    lineHeight: 20,
  },
});

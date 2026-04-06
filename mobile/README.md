# GlycoGuard Mobile

This is a React Native Android client for the local GlycoGuard API. It polls the existing `/watch/payload` endpoint on your machine, renders the risk card, and raises local Android notifications when risk escalates.

## What it connects to

Run the backend from the repo root:

```powershell
python -m glycoguard.cli serve
```

The mobile app reads from:

```text
GET http://<your-computer-lan-ip>:8000/watch/payload
GET http://<your-computer-lan-ip>:8000/watch/payload?patient_id=<patient-id>
```

Your phone must be on the same Wi-Fi network as your computer.

Do not use `localhost` on the phone. Use your computer LAN IP, for example:

```text
http://192.168.1.20:8000
```

## Expo setup

Expo SDK 54 maps to React Native 0.81 and React 19.1 in the official Expo SDK reference, and Expo's notifications docs state that local notifications remain available in Expo Go while remote push on Android requires a development build:

- https://docs.expo.dev/versions/latest/
- https://docs.expo.dev/versions/latest/sdk/notifications

## Install and run

From `mobile/`:

```powershell
npm install
npx expo start --tunnel
```

Then:

1. Install Expo Go on your Android phone.
2. Scan the QR code from `expo start`.
3. In the app, enter your computer LAN API URL.
4. Tap `Connect`.
5. Allow Android notifications.

## Optional `.env`

You can prefill the connection values:

```powershell
copy .env.example .env
```

Then edit:

```text
EXPO_PUBLIC_GLYCOGUARD_API_URL=http://192.168.1.20:8000
EXPO_PUBLIC_GLYCOGUARD_PATIENT_ID=
```

## Installable APK build

If you want an actual installable APK instead of Expo Go, use EAS:

```powershell
npm install -g eas-cli
eas login
eas build -p android --profile preview
```

That uses [eas.json](/C:/Users/Anshu%20Raj/Desktop/gluco/mobile/eas.json).

## Alert behavior

What works now:

- live card rendering from your local machine
- local Android notifications when risk escalates from `LOW -> MEDIUM/HIGH`
- local Android notifications when `buzz` turns on

Practical limitation:

- this is reliable while the app is open
- reliable background or terminated-app alerts will require FCM or a native Android foreground service

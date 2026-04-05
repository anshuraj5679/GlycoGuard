from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def replay_session(api_url: str, patient_id: str | None = None, speed: float = 10.0) -> None:
    start_payload: dict[str, Any] = {}
    if patient_id:
        start_payload["patient_id"] = patient_id
    session = _post_json(f"{api_url.rstrip('/')}/replay/start", start_payload)
    session_id = session["session_id"]
    delay_seconds = max(0.1, 5.0 / max(speed, 0.1))

    while True:
        query = urllib.parse.urlencode({"session_id": session_id})
        with urllib.request.urlopen(f"{api_url.rstrip('/')}/watch/payload?{query}") as response:
            watch = json.loads(response.read().decode("utf-8"))
        print(
            json.dumps(
                {
                    "patient_id": watch["patient_id"],
                    "glucose": watch["glucose"],
                    "trend": watch["trend"],
                    "risk": watch["risk"],
                    "reason": watch["reason"],
                    "forecast_30min": watch["forecast_30min"],
                    "status": watch["status"],
                },
                indent=2,
            )
        )
        if watch.get("status") == "completed":
            break
        time.sleep(delay_seconds)

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from glycoguard.service import get_service


def _build_api_command(python_executable: str, api_port: int) -> list[str]:
    return [
        python_executable,
        "-m",
        "uvicorn",
        "glycoguard.api.main:app",
        "--app-dir",
        "src",
        "--reload",
        "--port",
        str(api_port),
    ]


def _build_dashboard_command(python_executable: str, dashboard_port: int) -> list[str]:
    return [
        python_executable,
        "-m",
        "streamlit",
        "run",
        str(Path("src") / "glycoguard" / "dashboard" / "app.py"),
        "--server.port",
        str(dashboard_port),
    ]


def _launch(command: list[str], creationflags: int = 0) -> subprocess.Popen[bytes]:
    return subprocess.Popen(command, creationflags=creationflags)


def _terminate_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def serve(api_port: int = 8000, dashboard_port: int = 8501, separate_windows: bool = False) -> None:
    python_executable = sys.executable
    api_command = _build_api_command(python_executable, api_port)
    dashboard_command = _build_dashboard_command(python_executable, dashboard_port)

    creationflags = 0
    if separate_windows and os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE

    api_process = _launch(api_command, creationflags=creationflags)
    dashboard_process = _launch(dashboard_command, creationflags=creationflags)
    processes = [api_process, dashboard_process]

    print(
        json.dumps(
            {
                "status": "running",
                "api_url": f"http://127.0.0.1:{api_port}",
                "dashboard_url": f"http://127.0.0.1:{dashboard_port}",
                "api_pid": api_process.pid,
                "dashboard_pid": dashboard_process.pid,
                "separate_windows": bool(separate_windows and os.name == "nt"),
            },
            indent=2,
        )
    )
    print("Press Ctrl+C to stop both services.")

    try:
        while True:
            if any(process.poll() is not None for process in processes):
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        _terminate_processes(processes)

    exit_codes = {
        "api_exit_code": api_process.poll(),
        "dashboard_exit_code": dashboard_process.poll(),
    }
    if any(code not in (0, None, signal.SIGTERM if hasattr(signal, "SIGTERM") else None) for code in exit_codes.values()):
        raise SystemExit(json.dumps(exit_codes, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glycoguard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("patients", help="List loaded patients.")
    subparsers.add_parser("report", help="Print the default patient report summary.")

    serve_parser = subparsers.add_parser("serve", help="Run the API and dashboard together.")
    serve_parser.add_argument("--api-port", type=int, default=8000)
    serve_parser.add_argument("--dashboard-port", type=int, default=8501)
    serve_parser.add_argument("--separate-windows", action="store_true", help="Open the API and dashboard in separate console windows on Windows.")

    train_parser = subparsers.add_parser("train", help="Retrain the ensemble on loaded patients.")
    train_parser.add_argument("--persist", action="store_true", help="Save artifacts after retraining.")

    ingest_parser = subparsers.add_parser("ingest-bundle", help="Ingest a patient directory bundle.")
    ingest_parser.add_argument("bundle_dir", help="Directory containing cgm.csv and optional context files.")
    ingest_parser.add_argument("--patient-id", default=None)
    ingest_parser.add_argument("--retrain", action="store_true")
    ingest_parser.add_argument("--persist", action="store_true")

    save_parser = subparsers.add_parser("save-artifacts", help="Persist current model artifacts.")
    save_parser.add_argument("--directory", default=None)

    benchmark_parser = subparsers.add_parser("benchmark-ohio", help="Run the OhioT1DM benchmark flow.")
    benchmark_parser.add_argument("root_dir", help="OhioT1DM root containing training and testing folders.")
    benchmark_parser.add_argument("--max-forecast-points", type=int, default=500)

    federated_parser = subparsers.add_parser("federated-demo", help="Run a local federated learning demo.")
    federated_parser.add_argument("--rounds", type=int, default=3)
    federated_parser.add_argument("--min-clients", type=int, default=2)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        serve(
            api_port=args.api_port,
            dashboard_port=args.dashboard_port,
            separate_windows=args.separate_windows,
        )
        return

    service = get_service()

    if args.command == "patients":
        print(json.dumps(service.list_patients(), indent=2))
        return

    if args.command == "report":
        report = service.get_report()
        summary = {
            "patient_id": report["patient_id"],
            "current_glucose": report["current_glucose"],
            "risk_level": report["prediction"]["risk_level"],
            "hypo_probability": report["prediction"]["hypo_probability"],
            "predicted_glucose_30min": report["prediction"]["predicted_glucose_30min"],
        }
        print(json.dumps(summary, indent=2))
        return

    if args.command == "train":
        print(json.dumps(service.retrain(persist=args.persist), indent=2))
        return

    if args.command == "ingest-bundle":
        result = service.ingest_bundle(
            bundle_dir=args.bundle_dir,
            patient_id=args.patient_id,
            retrain=args.retrain,
            persist=args.persist,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "save-artifacts":
        print(json.dumps(service.save_artifacts(directory=args.directory), indent=2))
        return

    if args.command == "benchmark-ohio":
        print(
            json.dumps(
                service.benchmark_ohio(
                    root_dir=args.root_dir,
                    max_forecast_points=args.max_forecast_points,
                ),
                indent=2,
            )
        )
        return

    if args.command == "federated-demo":
        print(
            json.dumps(
                service.run_federated_demo(
                    rounds=args.rounds,
                    min_clients=args.min_clients,
                ),
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()

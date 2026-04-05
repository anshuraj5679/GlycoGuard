from __future__ import annotations

from glycoguard.cli import _build_api_command, _build_dashboard_command, build_parser


def test_serve_parser_accepts_ports_and_windows_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["serve", "--api-port", "9000", "--dashboard-port", "8601", "--separate-windows"])
    assert args.command == "serve"
    assert args.api_port == 9000
    assert args.dashboard_port == 8601
    assert args.separate_windows is True


def test_serve_commands_target_api_and_dashboard() -> None:
    api_command = _build_api_command("python", 8000)
    dashboard_command = _build_dashboard_command("python", 8501)

    assert api_command[:4] == ["python", "-m", "uvicorn", "glycoguard.api.main:app"]
    assert "--port" in api_command
    assert "8000" in api_command

    assert dashboard_command[:4] == ["python", "-m", "streamlit", "run"]
    assert dashboard_command[-2:] == ["--server.port", "8501"]

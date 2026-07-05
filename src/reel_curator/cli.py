from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from reel_curator.analyzer import analyze_library
from reel_curator.config import load_config
from reel_curator.exports import export_contact_sheet, export_csv, export_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local travel video curator")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Analyze the configured video folder")
    scan.add_argument("--input", type=Path, help="Video folder")
    scan.add_argument("--force", action="store_true", help="Ignore cached analyses")

    ui = sub.add_parser("ui", help="Launch Streamlit review UI")
    ui.add_argument("--port", type=int, default=8501)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "scan":
        config = load_config()
        if args.input:
            config.input_folder = args.input
        analyses = analyze_library(config, force=args.force)
        export_csv(analyses, config.export_dir / "metadata.csv")
        export_json(analyses, config.export_dir / "metadata.json")
        export_contact_sheet(analyses, config.export_dir / "contact_sheet.pdf")
        print(f"Analyzed {len(analyses)} videos. Exports written to {config.export_dir}")
        return 0

    if args.command == "ui":
        app_path = Path(__file__).parent / "ui" / "app.py"
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                str(args.port),
            ]
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

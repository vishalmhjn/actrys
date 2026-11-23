"""Command line interface: ``actrys {synthetic,sumo} CONFIG.toml [--set key=value ...]``."""

from __future__ import annotations

import argparse
import ast
import logging
import sys
import time

from actrys.config import (
    SumoCalibrationConfig,
    SyntheticCalibrationConfig,
    load_config,
    replace_path,
)


def _parse_override(text: str) -> tuple[str, object]:
    key, sep, raw = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {text!r}")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = {"true": True, "false": False}.get(raw.lower(), raw)
    return key.strip(), value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="actrys", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("synthetic", "calibrate the analytical (static assignment) scenario"),
        ("sumo", "calibrate a SUMO scenario"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", help="TOML configuration file")
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            type=_parse_override,
            metavar="KEY=VALUE",
            help="override a config value, e.g. demand.bagging_runs=1",
        )
        p.add_argument("--seed", type=int, default=None, help="random seed (overrides the config)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cls = SyntheticCalibrationConfig if args.command == "synthetic" else SumoCalibrationConfig
    cfg = load_config(cls, args.config)
    for key, value in args.overrides:
        cfg = _apply_override(cfg, key, value)
    if args.seed is not None:
        cfg = replace_path(cfg, "seed", args.seed)

    start = time.perf_counter()
    if args.command == "synthetic":
        from actrys.calibration.synthetic_pipeline import run_synthetic_calibration

        run_synthetic_calibration(cfg)
    else:
        from actrys.calibration.sumo_pipeline import run_sumo_calibration

        run_sumo_calibration(cfg)
    logging.getLogger("actrys").info("Finished in %.1f s", time.perf_counter() - start)
    return 0


def _apply_override(cfg, key: str, value):
    import dataclasses
    from pathlib import Path

    # coerce to the field's current type where that is unambiguous
    obj = cfg
    for part in key.split(".")[:-1]:
        obj = getattr(obj, part)
    name = key.split(".")[-1]
    if not dataclasses.is_dataclass(obj) or name not in {f.name for f in dataclasses.fields(obj)}:
        raise SystemExit(f"Unknown config key: {key}")
    current = getattr(obj, name)
    if isinstance(current, Path):
        value = Path(value)
    elif isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    elif isinstance(current, tuple) and isinstance(value, list):
        value = tuple(value)
    return replace_path(cfg, key, value)


if __name__ == "__main__":
    sys.exit(main())

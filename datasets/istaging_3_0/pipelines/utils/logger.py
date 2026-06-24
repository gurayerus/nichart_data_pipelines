"""
Shared logging setup for iSTAGING pipelines.

Usage in any pipeline script:
    import logging, sys
    from pathlib import Path
    _PIPELINES = Path(__file__).resolve().parents[N]   # N to reach pipelines/
    sys.path.insert(0, str(_PIPELINES / "utils"))
    from logger import setup_logger

    _log = logging.getLogger(__name__)

    def main():
        ...
        setup_logger(__name__, verbose=args.verbose,
                     log_dir=Path(args.log_dir) if args.log_dir else None)

Levels:
    Console : INFO (simple mode) or DEBUG (verbose mode).
    Log file: DEBUG and above — always written when log_dir is given.
              One file per run: <log_dir>/<script_name>_YYYYMMDD_HHMMSS.log
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

_FMT_CONSOLE         = "%(message)s"
_FMT_CONSOLE_VERBOSE = "%(levelname)s  %(message)s"
_FMT_FILE            = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_DATE_FMT            = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    verbose: bool = False,
    log_dir: Path | None = None,
) -> logging.Logger:
    """Configure and return the named logger.

    Calling this a second time replaces the handlers (safe to re-call).

    Args:
        name:     Logger name — use ``__name__`` from the calling module.
        verbose:  If True, console shows DEBUG messages with level prefix.
        log_dir:  Directory for the WARNING+ log file.  No file is created
                  when None.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(
        _FMT_CONSOLE_VERBOSE if verbose else _FMT_CONSOLE
    ))
    logger.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        # use only the leaf module name so filenames stay readable
        short_name = name.split(".")[-1]
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path   = log_dir / f"{short_name}_{ts}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FMT_FILE, datefmt=_DATE_FMT))
        logger.addHandler(fh)
        logger.info(f"Log: {log_path}")

    return logger

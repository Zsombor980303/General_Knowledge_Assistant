import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    # Egységes naplóformátum beállítása az egész alkalmazásra (print() helyett).
    # Minden sor elé odateszi az időbélyeget, a szintet (INFO/ERROR) és a modul nevét.
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    # Ezzel kér le saját névvel ellátott naplózót minden modul (pl. get_logger(__name__)).
    return logging.getLogger(name)
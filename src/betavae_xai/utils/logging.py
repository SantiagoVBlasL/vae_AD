# src/betavae_xai/utils/logging.py
import logging
import re
import warnings
from typing import Optional

class _CompactFormatter(logging.Formatter):
    """
    INFO sin prefijo; WARNING/ERROR/CRITICAL con nivel.
    Mantiene salida mucho más limpia en runs largos.
    """
    def __init__(self) -> None:
        super().__init__()
        self._fmt_info = logging.Formatter("%(message)s")
        self._fmt_other = logging.Formatter("[%(levelname)s] %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return self._fmt_info.format(record)
        return self._fmt_other.format(record)

class _FoldPrefixStripFilter(logging.Filter):
    """
    Elimina el prefijo 'Fold X/Y' al inicio del mensaje (y sus separadores),
    para evitar repetirlo en cada línea.
    """
    _re = re.compile(r"^(?P<indent>\s*)Fold\s+\d+/\d+(?:[\s:,-]+)")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        m = self._re.match(msg)
        if m:
            record.msg = m.group("indent") + msg[m.end():]
            record.args = ()
        return True

def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """
    Configura el logger root con formato compacto y filtros, y retorna
    un logger específico con el nombre dado (usualmente __name__).
    """
    # Root logger: configuración única, incluso en notebooks/handlers previos.
    logging.basicConfig(level=logging.INFO, force=True)

    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    
    for h in root.handlers:
        h.setLevel(logging.INFO)
        h.setFormatter(_CompactFormatter())
        # Evitamos duplicar filtros si se llama múltiples veces
        if not any(isinstance(f, _FoldPrefixStripFilter) for f in h.filters):
            h.addFilter(_FoldPrefixStripFilter())

    # Reducir ruido de librerías
    for noisy in ["lightgbm", "optuna", "sklearn", "xgboost"]:
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # warnings globales: una sola vez
    warnings.filterwarnings("ignore")
    warnings.filterwarnings(
        "ignore",
        message=".*X does not have valid feature names.*",
        category=UserWarning,
        module="sklearn.utils.validation",
    )
    
    return logging.getLogger(name)
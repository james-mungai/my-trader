from dataclasses import dataclass

from futures_lab.config import Settings


@dataclass
class NautilusAvailability:
    installed: bool
    reason: str


def check_nautilus() -> NautilusAvailability:
    try:
        import nautilus_trader  # noqa: F401
    except Exception as exc:
        return NautilusAvailability(
            installed=False,
            reason=(
                "NautilusTrader is not installed in this environment. Install with "
                "`pip install -e '.[nautilus]'` after creating a Python 3.11+ virtualenv. "
                f"Import error: {exc}"
            ),
        )
    return NautilusAvailability(installed=True, reason="NautilusTrader import succeeded.")


def nautilus_symbol(settings: Settings) -> str:
    # Nautilus distinguishes the Binance perpetual from the spot pair with `-PERP`.
    return f"{settings.symbol.upper()}-PERP"


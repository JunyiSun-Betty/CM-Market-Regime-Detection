"""Small, backend-safe client for the CryptoMatter market-data API."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://backend.crypto-matter.com/api"
ALLOWED_COLUMNS = {"Close", "Market_cap", "Trade_volume"}
ENV_FILE = Path(__file__).with_name(".env")


class CryptoMatterError(RuntimeError):
    """Raised when CryptoMatter returns an error or cannot be reached."""


def _get_api_key() -> str | None:
    """Read CM_API_KEY from the environment, falling back to the local .env file."""
    if api_key := os.environ.get("CM_API_KEY"):
        return api_key

    try:
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "CM_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            return value or None
    return None


def get_series(coin: str, column: str = "Close", *, timeout: int = 30) -> list[dict]:
    """Return the complete daily history for one coin and one column.

    The API key is read only from the CM_API_KEY environment variable.
    """
    if not coin:
        raise ValueError("coin must not be empty")
    if column not in ALLOWED_COLUMNS:
        raise ValueError(f"column must be one of: {', '.join(sorted(ALLOWED_COLUMNS))}")

    api_key = _get_api_key()
    if not api_key:
        raise CryptoMatterError("CM_API_KEY is not set in the environment or .env file")

    query = urlencode({"coin": coin, "column": column})
    request = Request(
        f"{BASE_URL}/coins/marketdata?{query}",
        headers={"X-API-Key": api_key, "Accept": "application/json"},
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        raise CryptoMatterError(f"CryptoMatter returned HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise CryptoMatterError(f"Could not reach CryptoMatter: {exc.reason}") from exc

    if not isinstance(payload, list):
        raise CryptoMatterError("Unexpected response: expected a JSON array")
    return payload


def get_series_as_rows(coin: str, column: str = "Close") -> list[tuple[date, float | None]]:
    """Return validated (date, value) rows, preserving API order."""
    rows = get_series(coin, column)
    result: list[tuple[date, float | None]] = []
    for row in rows:
        try:
            day = date.fromisoformat(row["CreationDate"])
            value = row.get(column)
        except (KeyError, TypeError, ValueError) as exc:
            raise CryptoMatterError(f"Unexpected row format: {row!r}") from exc
        if value is not None and not isinstance(value, (int, float)):
            raise CryptoMatterError(f"Unexpected value in {column}: {value!r}")
        result.append((day, value))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch CryptoMatter daily market data")
    parser.add_argument("coin", nargs="?", default="bitcoin")
    parser.add_argument("--column", default="Close", choices=sorted(ALLOWED_COLUMNS))
    args = parser.parse_args()

    rows = get_series_as_rows(args.coin, args.column)
    print(f"Fetched {len(rows)} rows for {args.coin}/{args.column}")
    for day, value in rows[-5:]:
        print(f"{day.isoformat()}\t{value}")

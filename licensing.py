"""
loom.licensing
------------------
Freemium gating: the CLI is free for individual use, with two limits —
a daily run cap and a single-attempt repair loop. A "Team" license lifts
both and unlocks the audit/cost report's export (see loom/report.py).

How license keys work (offline, no server required):
  - Anthropic-style CLIs can't phone home for every invocation without
    annoying offline/CI users, so verification is fully offline: a license
    is a JSON payload (org, tier, seats, issued/expires) signed with
    Ed25519 and base64-encoded into one string.
  - This file embeds only the PUBLIC key — safe to publish, since a public
    key can verify a signature but cannot forge a new one. Publishing this
    repo does NOT let anyone mint their own valid license.
  - The matching PRIVATE key is never in this repo. It lives only with
    whoever is selling licenses (see admin/generate_license.py, and the
    big warning in admin/README.md) and is used, offline, to sign a new
    license string each time someone pays.

Honest limitation: this is client-side verification, not a phone-home
license server — a determined person could patch out the check in Python
source. That's normal and accepted for this class of B2B tool (the
signature exists to make "here's your license key" a real, unforgeable
artifact for legitimate customers, not to defeat piracy outright). If you
outgrow that trust model, this file is the one place to swap in a
server-side check.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Public half of the Ed25519 keypair used to sign license keys.
# Safe to be here, safe to be public — see module docstring.
PUBLIC_KEY_B64 = "k9ybqXzV32xoYrjMo8cvW5JC4lU0FZiXAePUzJUVccw="

LICENSE_PATH = Path.home() / ".loom" / "license.json"
USAGE_PATH = Path.home() / ".loom" / "usage.json"

FREE_TIER_DAILY_RUNS = 5
FREE_TIER_MAX_REPAIRS = 1


class InvalidLicenseError(Exception):
    pass


@dataclass
class License:
    org: str
    tier: str              # "team"
    seats: int
    issued: str
    expires: Optional[str]
    raw: str                # the original activation string, so it can be re-verified from disk

    def is_expired(self) -> bool:
        if not self.expires:
            return False
        return time.time() > time.mktime(time.strptime(self.expires, "%Y-%m-%d"))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


def verify_license_string(license_str: str) -> License:
    """Decode + verify a `loom license activate` string. Raises
    InvalidLicenseError with a human-readable reason on any failure —
    bad base64, bad JSON, wrong signature, or missing fields."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as e:
        raise InvalidLicenseError(
            "License verification needs the 'cryptography' package. Install with: "
            "pip install -e \".[license]\""
        ) from e

    try:
        payload_b64, sig_b64 = license_str.strip().split(".", 1)
        payload_bytes = _b64url_decode(payload_b64)
        sig_bytes = _b64url_decode(sig_b64)
    except Exception as e:
        raise InvalidLicenseError(f"Malformed license string: {e}") from e

    public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(PUBLIC_KEY_B64))
    try:
        public_key.verify(sig_bytes, payload_bytes)
    except InvalidSignature:
        raise InvalidLicenseError("Signature does not match — this license key is invalid or was tampered with.")

    try:
        data = json.loads(payload_bytes)
        org, tier, seats, issued = data["org"], data["tier"], data["seats"], data["issued"]
        expires = data.get("expires")
    except (json.JSONDecodeError, KeyError) as e:
        raise InvalidLicenseError(f"License payload is missing required fields: {e}") from e

    lic = License(org=org, tier=tier, seats=seats, issued=issued, expires=expires, raw=license_str)
    if lic.is_expired():
        raise InvalidLicenseError(f"This license expired on {expires}.")
    return lic


class LicenseStore:
    def __init__(self, path: Path = LICENSE_PATH):
        self.path = path

    def activate(self, license_str: str) -> License:
        lic = verify_license_string(license_str)  # raises before writing anything if invalid
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"raw": license_str}))
        return lic

    def current(self) -> Optional[License]:
        """Always re-verifies the signature on the stored string rather than
        trusting cached fields — a hand-edited license.json (e.g. someone
        changing tier to 'team' by hand) fails signature verification and
        is treated as no license at all."""
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text())
            return verify_license_string(data["raw"])
        except (json.JSONDecodeError, KeyError, InvalidLicenseError):
            return None

    def deactivate(self) -> None:
        if self.path.exists():
            self.path.unlink()


def is_team_licensed(store: Optional[LicenseStore] = None) -> bool:
    store = store or LicenseStore()
    lic = store.current()
    return lic is not None and lic.tier == "team"


class UsageLimiter:
    """Tracks `loom run` invocation timestamps to enforce the free-tier
    daily cap. Global (not per-project) — the limit is per install, matching
    how the license itself is global."""

    def __init__(self, path: Path = USAGE_PATH):
        self.path = path

    def _load(self) -> list[float]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return []

    def _save(self, timestamps: list[float]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(timestamps))

    def runs_in_last_24h(self) -> int:
        cutoff = time.time() - 86400
        return len([t for t in self._load() if t > cutoff])

    def record_run(self) -> None:
        cutoff = time.time() - 86400
        timestamps = [t for t in self._load() if t > cutoff]
        timestamps.append(time.time())
        self._save(timestamps)


def check_run_allowed(license_store: Optional[LicenseStore] = None,
                       usage: Optional[UsageLimiter] = None) -> tuple[bool, str, int]:
    """Returns (allowed, message_if_blocked, effective_max_repairs)."""
    license_store = license_store or LicenseStore()
    usage = usage or UsageLimiter()

    if is_team_licensed(license_store):
        return True, "", 999  # effectively unlimited repair attempts

    used = usage.runs_in_last_24h()
    if used >= FREE_TIER_DAILY_RUNS:
        return False, (
            f"Free tier is limited to {FREE_TIER_DAILY_RUNS} runs per 24 hours "
            f"(you've used {used}). Upgrade with: loom license activate <key>"
        ), FREE_TIER_MAX_REPAIRS
    return True, "", FREE_TIER_MAX_REPAIRS

#!/usr/bin/env python3
"""
admin/generate_license.py
----------------------------
YOU run this — your customers never see or need it. It signs a new
license key that a customer activates with `loom license activate <key>`.

This script does NOT contain your private signing key. It reads it from
an environment variable or a file you point it at, specifically so this
script is safe to keep in the same (potentially public) repo as the rest
of Loom — there is no secret in this file, only the logic that USES one.

Usage:
    export LOOM_SIGNING_KEY=<base64 private key>
    python admin/generate_license.py --org "Acme Inc" --seats 10 --tier team --expires 2027-08-08

    # or, keeping the key in a local file instead of an env var:
    python admin/generate_license.py --org "Acme Inc" --seats 10 --tier team \\
        --signing-key-file ~/.loom-signing-key.txt

Output is the license string to send to the customer — send it directly
(email, invoice attachment, whatever your sales process is). There's
nothing sensitive in the license string itself; it only encodes org name,
seat count, tier, and expiry, signed so it can't be forged or edited.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def load_signing_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key_b64 = os.environ.get("LOOM_SIGNING_KEY")
    key_file = None
    if "--signing-key-file" in sys.argv:
        key_file = Path(sys.argv[sys.argv.index("--signing-key-file") + 1])

    if key_file:
        key_b64 = key_file.read_text().strip()
    if not key_b64:
        sys.exit(
            "No signing key found. Set LOOM_SIGNING_KEY or pass --signing-key-file.\n"
            "This is the PRIVATE key from when you first set up licensing — keep it "
            "somewhere private (a password manager, not a git repo)."
        )
    return Ed25519PrivateKey.from_private_bytes(base64.b64decode(key_b64))


def main():
    parser = argparse.ArgumentParser(description="Sign a new Loom license key.")
    parser.add_argument("--org", required=True, help="Customer / organization name")
    parser.add_argument("--seats", type=int, default=1)
    parser.add_argument("--tier", default="team", choices=["team"])
    parser.add_argument("--expires", default=None, help="YYYY-MM-DD, omit for a perpetual license")
    parser.add_argument("--signing-key-file", help="Path to a file containing the base64 private key")
    args = parser.parse_args()

    private_key = load_signing_key()

    import datetime
    payload = {
        "org": args.org,
        "tier": args.tier,
        "seats": args.seats,
        "issued": datetime.date.today().isoformat(),
        "expires": args.expires,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = private_key.sign(payload_bytes)

    license_str = f"{b64url_encode(payload_bytes)}.{b64url_encode(signature)}"
    print("\nLicense key (send this to the customer):\n")
    print(license_str)
    print(f"\n({payload['org']}, {payload['seats']} seats, tier={payload['tier']}, "
          f"expires={payload['expires'] or 'never'})")


if __name__ == "__main__":
    main()

# Admin tools — for you, not your customers

`generate_license.py` signs new license keys for people who pay for a Team
license. It's safe to keep in this repo (even a public one) because it
contains no secret — it *reads* your private signing key from an
environment variable or a file, it never stores one.

**Your private signing key is NOT in this repo anywhere.** It was
generated once, and only exists wherever you personally saved it.

## Rules for the private key

- **Never commit it to git, any repo, public or private.**
- **Never paste it into a chat, ticket, or Slack message that others can see.**
- Store it in a password manager, or a local file outside any git repo
  (e.g. `~/.loom-signing-key.txt`, which is a location on your machine,
  not this project folder).
- If you ever suspect it leaked: anyone who has it can mint unlimited
  valid "Team" licenses for free. There's no revocation mechanism in this
  v1 (see the honest limitation noted in `loom/licensing.py`) — the fix
  is a fresh keypair, a new public key shipped in the next Loom release,
  and re-issuing licenses to your real customers.

## Generating your first license (for yourself, to test)

```bash
export LOOM_SIGNING_KEY="<the private key you were given when this was set up>"
python admin/generate_license.py --org "Test Co" --seats 1 --tier team
```

Copy the printed license string, then in a normal terminal (not this admin
one):

```bash
loom license activate <the license string>
loom license status
```

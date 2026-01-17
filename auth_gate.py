# auth_gate.py
import os, time, hmac, hashlib, base64
import streamlit as st

def _b64url_decode(s: str) -> bytes:
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode("utf-8"))

def verify_token(token: str, secret: str) -> tuple[bool, str]:
    try:
        part_payload, part_sig = token.split(".", 1)
        payload = _b64url_decode(part_payload).decode("utf-8")
        sig = _b64url_decode(part_sig).decode("utf-8")

        email, exp_str = payload.split("|", 1)
        exp = int(exp_str)

        if time.time() > exp:
            return (False, "")

        expected = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected, sig):
            return (False, "")

        return (True, email)
    except Exception:
        return (False, "")

def require_admin_token() -> str:
    secret = os.environ.get("OS_TOKEN_SECRET_ADMIN", "").strip()
    token = st.query_params.get("t", "")

    if not secret or not token:
        st.stop()

    ok, authed_email = verify_token(token, secret)
    if not ok:
        st.stop()

    allow = set(
        e.strip().lower()
        for e in os.environ.get("ADMIN_EMAILS", "").split(",")
        if e.strip()
    )
    if allow and authed_email.strip().lower() not in allow:
        st.stop()

    return authed_email

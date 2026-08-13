from __future__ import annotations

import hmac
import os

import streamlit as st


def _parse_keys(raw: str | None) -> list[str]:
    """
    Same parsing logic as auth_middleware.py's _parse_keys -- comma-
    separated, empty/None becomes an empty list, not a list containing
    one empty string. That distinction matters here specifically:
    without it, an empty text input could accidentally match an
    unconfigured ADMIN_API_KEYS, defeating the whole gate.
    """
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def require_admin() -> None:
    """
    Call at the very top of any Streamlit page that should be
    admin-only, before any real page content. Streamlit executes a
    script top-to-bottom on every rerun, so st.stop() below is what
    actually prevents the rest of the calling page from running at all
    while unauthenticated -- without it, this function returning early
    would do nothing, the page's own content would still execute right
    after it regardless.

    Reads the SAME ADMIN_API_KEYS env var auth_middleware.py already
    reads on the FastAPI side, not a second, separately-configured
    credential system -- one source of truth for what counts as a valid
    admin key, same reasoning as everywhere else RBAC applies in this
    project. Requires ADMIN_API_KEYS to actually be passed into the
    streamlit service's environment in docker-compose.yml (it currently
    is not) for this to have anything real to check against.
    """
    if st.session_state.get("is_admin_authenticated", False):
        return

    st.title("Admin access required")
    st.caption("This page is restricted to admin API keys.")

    provided_key = st.text_input("Admin API key", type="password", key="admin_key_input")

    if st.button("Log in"):
        valid_keys = _parse_keys(os.environ.get("ADMIN_API_KEYS"))

        # Fail-closed, same as the FastAPI-side default elsewhere in
        # this project: no configured keys means no possible match, not
        # an accidental bypass. hmac.compare_digest for constant-time
        # comparison, same discipline as auth_middleware.py, checked
        # against each valid key individually since compare_digest
        # doesn't support "is this in a list" directly.
        is_valid = any(
            hmac.compare_digest(provided_key, valid_key) for valid_key in valid_keys
        )

        if is_valid and provided_key:
            st.session_state["is_admin_authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid admin key.")

    st.stop()

"""Snowpark session, only ever created when SNOWFLAKE_MODE=snowflake."""

import subprocess

import config

_session = None


def _ensure_local_network_ca_trusted() -> None:
    """Adds CAs trusted by the macOS keychain to the connector's CA bundle (for TLS-inspecting proxies)."""
    import os
    import certifi

    combined_path = os.path.join(config.ROOT_DIR.as_posix(), ".local_ca_bundle.pem")
    if os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE"):
        return  # caller already configured a trust bundle explicitly

    try:
        result = subprocess.run(
            ["security", "find-certificate", "-a", "-p", "/Library/Keychains/System.keychain"],
            capture_output=True, text=True, timeout=10,
        )
        system_certs = result.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return  # not macOS, or `security` unavailable — plain certifi is used

    if "BEGIN CERTIFICATE" not in system_certs:
        return

    with open(certifi.where()) as f:
        certifi_bundle = f.read()

    with open(combined_path, "w") as f:
        f.write(certifi_bundle)
        f.write("\n")
        f.write(system_certs)

    os.environ["REQUESTS_CA_BUNDLE"] = combined_path
    os.environ["SSL_CERT_FILE"] = combined_path


def set_session(session) -> None:
    """Use an existing session — inside a Snowflake stored procedure, the one
    Snowflake passes in, rather than connecting again with .env credentials."""
    global _session
    _session = session


def get_session():
    global _session
    if config.SNOWFLAKE_MODE != "snowflake":
        raise RuntimeError(
            "SNOWFLAKE_MODE is 'local' — no Snowpark session needed. "
            "Set SNOWFLAKE_MODE=snowflake in .env once you have a trial account."
        )
    if _session is None:
        _ensure_local_network_ca_trusted()
        from snowflake.snowpark import Session

        missing = [k for k, v in config.SNOWFLAKE_CONNECTION_PARAMS.items() if not v and k != "role"]
        if missing:
            raise RuntimeError(f"Missing Snowflake connection params: {missing}")
        _session = Session.builder.configs(config.SNOWFLAKE_CONNECTION_PARAMS).create()
    return _session

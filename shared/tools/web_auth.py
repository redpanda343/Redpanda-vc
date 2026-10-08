import ipaddress
import os
import secrets


def _is_loopback(server_name):
    normalized_name = server_name.strip().lower()
    try:
        return ipaddress.ip_address(normalized_name).is_loopback
    except ValueError:
        return normalized_name == "localhost"


def build_launch_auth(server_name, share):
    configured_username = os.environ.get("APPLIO_AUTH_USERNAME")
    configured_password = os.environ.get("APPLIO_AUTH_PASSWORD")
    credentials_requested = (
        configured_username is not None or configured_password is not None
    )

    if not share and _is_loopback(server_name) and not credentials_requested:
        return None

    username = (configured_username or "applio").strip()
    password = configured_password or secrets.token_urlsafe(24)

    if not username:
        raise ValueError("APPLIO_AUTH_USERNAME cannot be empty.")
    if configured_password and len(configured_password) < 16:
        raise ValueError("APPLIO_AUTH_PASSWORD must contain at least 16 characters.")

    print("Authentication is enabled because the WebUI is network-accessible.")
    print(f"Username: {username}")
    if configured_password:
        print("Password: loaded from APPLIO_AUTH_PASSWORD")
    else:
        print(f"Generated password: {password}")
        print(
            "Set APPLIO_AUTH_USERNAME and APPLIO_AUTH_PASSWORD for fixed credentials."
        )

    return username, password

import inspect
import logging
from urllib.parse import urlsplit

import httpx
from fastapi import Request, Response

from shared.tools.launch_tensorboard import get_tb_url

MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 128 * 1024 * 1024
REQUEST_HEADERS = {
    "accept",
    "accept-language",
    "content-type",
    "if-modified-since",
    "if-none-match",
    "range",
    "user-agent",
}
RESPONSE_HEADERS = {
    "accept-ranges",
    "content-disposition",
    "content-range",
    "content-type",
    "etag",
    "last-modified",
}
logger = logging.getLogger(__name__)


async def _current_user(app, request):
    if app.auth is None and app.auth_dependency is None:
        return "local"
    if app.auth_dependency is not None:
        user = app.auth_dependency(request)
        if inspect.isawaitable(user):
            user = await user
        return user
    token = request.cookies.get(
        f"access-token-{app.cookie_id}"
    ) or request.cookies.get(f"access-token-unsecure-{app.cookie_id}")
    return app.tokens.get(token)


def _same_origin(request):
    if request.method != "POST":
        return True
    origin = request.headers.get("origin")
    host = request.headers.get("host")
    if not origin or not host:
        return False
    parsed = urlsplit(origin)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.casefold() == host.casefold()
    )


def _target_url(path, query):
    base_url = get_tb_url()
    if not base_url:
        return None
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        return None
    if "\\" in path or "\x00" in path:
        return None
    if any(part in {".", ".."} for part in path.split("/")):
        return None
    target = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    if query:
        target = f"{target}?{query}"
    return target


def _redirect_location(location, base_url):
    if not location:
        return None
    parsed = urlsplit(location)
    if not parsed.scheme and not parsed.netloc:
        if location.startswith("//"):
            return None
        return location
    base = urlsplit(base_url)
    if (
        parsed.scheme != base.scheme
        or parsed.hostname != base.hostname
        or parsed.port != base.port
    ):
        return None
    rewritten = parsed.path
    if parsed.query:
        rewritten = f"{rewritten}?{parsed.query}"
    return rewritten


def _proxy_headers(response, base_url):
    headers = {
        "Cache-Control": "no-store",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Permissions-Policy": "camera=(), geolocation=(), microphone=()",
        "Referrer-Policy": "same-origin",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
    }
    content_security_policy = response.headers.get("content-security-policy")
    if content_security_policy:
        headers["Content-Security-Policy"] = (
            f"{content_security_policy}; frame-ancestors 'self'"
        )
    else:
        headers["Content-Security-Policy"] = "frame-ancestors 'self'"
    for name in RESPONSE_HEADERS:
        value = response.headers.get(name)
        if value:
            headers[name] = value
    location = _redirect_location(response.headers.get("location"), base_url)
    if location:
        headers["Location"] = location
    return headers


def register_tensorboard_proxy(app):
    first_new_route = len(app.router.routes)

    @app.api_route(
        "/tensorboard/{path:path}", methods=["GET", "HEAD", "POST"]
    )
    @app.api_route("/tensorboard", methods=["GET", "HEAD", "POST"])
    async def tensorboard_proxy(request: Request, path: str = ""):
        if not await _current_user(app, request):
            return Response(
                "Authentication required",
                status_code=401,
                headers={"Cache-Control": "no-store"},
            )
        if not _same_origin(request):
            return Response(
                "Cross-origin request denied",
                status_code=403,
                headers={"Cache-Control": "no-store"},
            )

        target_url = _target_url(path, request.url.query)
        if not target_url:
            return Response(
                "TensorBoard is not available",
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )
        base_url = get_tb_url()
        if not base_url:
            return Response(
                "TensorBoard is not available",
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )

        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                return Response(
                    "Request is too large",
                    status_code=413,
                    headers={"Cache-Control": "no-store"},
                )
            body.extend(chunk)

        request_headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() in REQUEST_HEADERS
        }

        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=httpx.Timeout(60, connect=5),
                trust_env=False,
            ) as client:
                async with client.stream(
                    request.method,
                    target_url,
                    headers=request_headers,
                    content=bytes(body),
                ) as response:
                    response_body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(response_body) + len(chunk) > MAX_RESPONSE_BYTES:
                            return Response(
                                "TensorBoard response is too large",
                                status_code=502,
                                headers={"Cache-Control": "no-store"},
                            )
                        response_body.extend(chunk)
                    status_code = response.status_code
                    response_headers = _proxy_headers(response, base_url)
                    unsafe_redirect = response.is_redirect and response.headers.get(
                        "location"
                    ) and not response_headers.get("Location")
        except httpx.HTTPError as error:
            logger.warning("TensorBoard proxy request failed: %s", error)
            return Response(
                "TensorBoard is not available",
                status_code=502,
                headers={"Cache-Control": "no-store"},
            )

        if unsafe_redirect:
            return Response(
                "TensorBoard returned an unsafe redirect",
                status_code=502,
                headers={"Cache-Control": "no-store"},
            )

        return Response(
            content=bytes(response_body),
            status_code=status_code,
            headers=response_headers,
        )

    new_routes = app.router.routes[first_new_route:]
    del app.router.routes[first_new_route:]
    app.router.routes[0:0] = new_routes

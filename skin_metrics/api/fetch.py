"""Fetch and decode a remote image for the HTTP API.

The API takes a user-supplied URL and the *server* performs the request, so
this module is the SSRF boundary. Every fetch is checked for:

* scheme allow-list (``http`` / ``https`` only),
* target IP reachability class (no loopback / private / link-local / reserved),
* redirect hops (each hop re-validated, bounded count),
* response size (streamed with a hard byte cap) and decoded pixel count.

Known residual risk: validation resolves DNS, then httpx resolves it again when
connecting, so a rebinding attacker with sub-second TTL control could slip
between the two. Deployments that treat that as in-scope should egress through
a proxy with its own allow-list instead of relying on ``allow_private_hosts``.
"""

from __future__ import annotations

import ipaddress
import socket
from io import BytesIO
from urllib.parse import urljoin, urlsplit

import numpy as np

from .settings import ApiSettings

_ALLOWED_SCHEMES = ("http", "https")
_DEFAULT_PORTS = {"http": 80, "https": 443}
_REDIRECT_CODES = (301, 302, 303, 307, 308)


class ImageFetchError(Exception):
    """A fetch/decode failure that maps onto an HTTP response.

    Attributes
    ----------
    code : str
        Stable machine-readable error code (e.g. ``"blocked_host"``).
    status_code : int
        HTTP status the API should return.
    """

    def __init__(self, message: str, *, code: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


def _resolve_host(host: str, port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve ``host`` to IP addresses.

    Parameters
    ----------
    host : str
        Hostname or IP literal.
    port : int
        Port used for the lookup.

    Returns
    -------
    list of ipaddress.IPv4Address or ipaddress.IPv6Address
        Every address the host resolves to.

    Raises
    ------
    ImageFetchError
        If the name cannot be resolved.
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ImageFetchError(
            f"Could not resolve host '{host}'.", code="dns_error", status_code=400
        ) from exc
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def _is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return ``True`` for addresses the server must not be pointed at."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_url(url: str, *, allow_private_hosts: bool = False) -> None:
    """Validate a user-supplied image URL against the SSRF rules.

    Parameters
    ----------
    url : str
        Absolute ``http``/``https`` URL.
    allow_private_hosts : bool, optional
        Skip the IP reachability check (development only).

    Raises
    ------
    ImageFetchError
        If the scheme, host, or resolved address is not allowed.
    """
    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise ImageFetchError(
            f"Unsupported URL scheme '{parts.scheme}'; use http or https.",
            code="invalid_scheme",
        )
    host = parts.hostname
    if not host:
        raise ImageFetchError("URL has no host.", code="invalid_url")
    if allow_private_hosts:
        return

    port = parts.port or _DEFAULT_PORTS[parts.scheme]
    for ip in _resolve_host(host, port):
        if _is_blocked(ip):
            raise ImageFetchError(
                f"Host '{host}' resolves to a non-public address ({ip}).",
                code="blocked_host",
                status_code=403,
            )


def decode_image(
    data: bytes, *, max_pixels: int, analysis_max_pixels: int | None = None
) -> np.ndarray:
    """Decode image bytes into an RGB ``uint8`` array.

    EXIF orientation is applied, palette/grayscale/alpha inputs are converted to
    3-channel RGB.

    Parameters
    ----------
    data : bytes
        Encoded image (JPEG/PNG/WebP/...).
    max_pixels : int
        Reject images whose ``width * height`` exceeds this.
    analysis_max_pixels : int, optional
        Downscale (never upscale) the decoded image to fit this budget. Left at
        ``None`` the image is returned at native size. See
        :attr:`~skin_metrics.api.settings.ApiSettings.analysis_max_pixels` for
        why the API sets one.

    Returns
    -------
    numpy.ndarray
        sRGB image, shape ``(H, W, 3)``, dtype ``uint8``.

    Raises
    ------
    ImageFetchError
        If the bytes are not a decodable image or the image is too large.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(BytesIO(data)) as im:
            width, height = im.size
            if width * height > max_pixels:
                raise ImageFetchError(
                    f"Image is {width}x{height} ({width * height} px); the limit is "
                    f"{max_pixels} px.",
                    code="image_too_large",
                    status_code=413,
                )
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            # exif_transpose may have swapped the axes, so re-read the size.
            if analysis_max_pixels is not None and im.width * im.height > analysis_max_pixels:
                scale = (analysis_max_pixels / (im.width * im.height)) ** 0.5
                target = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
                im = im.resize(target, Image.LANCZOS)
            arr = np.asarray(im, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageFetchError(
            "Could not decode the downloaded bytes as an image.",
            code="decode_error",
        ) from exc

    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        raise ImageFetchError("Decoded image is not a 3-channel RGB image.", code="decode_error")
    return arr


async def fetch_image(url: str, settings: ApiSettings, *, client=None) -> tuple[np.ndarray, dict]:
    """Download an image URL and decode it, enforcing the safety limits.

    Redirects are followed manually so that every hop passes
    :func:`validate_url`; the body is streamed and aborted past
    ``settings.max_bytes``.

    Parameters
    ----------
    url : str
        Image URL supplied by the client.
    settings : ApiSettings
        Limits (timeout, byte/pixel caps, redirect budget, host policy).
    client : httpx.AsyncClient, optional
        Reused client. A short-lived one is created when omitted.

    Returns
    -------
    tuple of (numpy.ndarray, dict)
        The decoded RGB image and fetch metadata (``final_url``,
        ``content_type``, ``bytes``, ``width``, ``height``).

    Raises
    ------
    ImageFetchError
        On a blocked target, transport failure, non-2xx status, oversized body,
        or undecodable content.
    """
    import httpx

    if client is None:
        async with httpx.AsyncClient(
            timeout=settings.fetch_timeout, follow_redirects=False
        ) as owned:
            return await fetch_image(url, settings, client=owned)

    current = url
    for _ in range(settings.max_redirects + 1):
        validate_url(current, allow_private_hosts=settings.allow_private_hosts)
        try:
            async with client.stream("GET", current, timeout=settings.fetch_timeout) as response:
                if response.status_code in _REDIRECT_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise ImageFetchError(
                            "Redirect response without a Location header.",
                            code="bad_redirect",
                            status_code=502,
                        )
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise ImageFetchError(
                        f"Image URL returned HTTP {response.status_code}.",
                        code="upstream_error",
                        status_code=502,
                    )

                declared = response.headers.get("content-length")
                if declared is not None and declared.isdigit():
                    if int(declared) > settings.max_bytes:
                        raise ImageFetchError(
                            f"Image is {declared} bytes; the limit is {settings.max_bytes}.",
                            code="image_too_large",
                            status_code=413,
                        )

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > settings.max_bytes:
                        raise ImageFetchError(
                            f"Image exceeds the {settings.max_bytes} byte limit.",
                            code="image_too_large",
                            status_code=413,
                        )
                    chunks.append(chunk)
                content_type = response.headers.get("content-type", "")
                final_url = str(response.url)
        except httpx.TimeoutException as exc:
            raise ImageFetchError(
                f"Timed out after {settings.fetch_timeout}s fetching the image.",
                code="fetch_timeout",
                status_code=504,
            ) from exc
        except httpx.HTTPError as exc:
            raise ImageFetchError(
                f"Could not fetch the image URL: {exc}", code="fetch_error", status_code=502
            ) from exc

        data = b"".join(chunks)
        if not data:
            raise ImageFetchError("Image URL returned an empty body.", code="empty_body")

        image = decode_image(
            data,
            max_pixels=settings.max_pixels,
            analysis_max_pixels=settings.analysis_max_pixels,
        )
        meta = {
            "final_url": final_url,
            "content_type": content_type.split(";")[0].strip() or None,
            "bytes": len(data),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }
        return image, meta

    raise ImageFetchError(
        f"Exceeded {settings.max_redirects} redirects.", code="too_many_redirects", status_code=502
    )

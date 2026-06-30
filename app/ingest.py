"""Server-side fetch of a hosted video by URL (the notebook's INPUT_URLS/fetch_url,
adapted for a multi-tenant API: scheme allowlist, size cap, and a private/loopback
IP guard against SSRF.

Note: the IP check below resolves the hostname once before connecting; it doesn't
pin that address for the actual request, so it doesn't fully close a DNS-rebinding
race. Good enough for trusted/internal callers; tighten with a pinned-socket
connector before exposing this to untrusted public clients.
"""

import ipaddress
import os
import socket
import tempfile
import urllib.request
from urllib.parse import urlparse

from fastapi import HTTPException

from . import config


def _assert_public_host(hostname: str) -> None:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise HTTPException(400, f"could not resolve host '{hostname}'")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise HTTPException(400, f"refusing to fetch from non-public address '{ip}'")


def download_video(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in config.ALLOWED_DOWNLOAD_SCHEMES:
        raise HTTPException(400, f"unsupported URL scheme '{parsed.scheme}'")
    if not parsed.hostname:
        raise HTTPException(400, "URL is missing a host")
    _assert_public_host(parsed.hostname)

    suffix = os.path.splitext(parsed.path)[1].lower()
    if suffix not in config.ALLOWED_SUFFIXES:
        suffix = ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="swarm_fetch_")

    req = urllib.request.Request(url, headers={"User-Agent": "swarm-audiences/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=config.DOWNLOAD_TIMEOUT_SECONDS) as resp:
            with os.fdopen(fd, "wb") as out:
                size = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > config.MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "downloaded file too large")
                    out.write(chunk)
    except HTTPException:
        os.remove(path)
        raise
    except Exception as e:
        os.remove(path)
        raise HTTPException(400, f"failed to download URL: {type(e).__name__}: {e}")
    return path

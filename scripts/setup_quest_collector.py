#!/usr/bin/env python3
"""Prepare a local CloudXR collector without starting servers or Isaac Sim.

Uses Python's standard library, curl, and openssl. Machine-specific files stay
in .external/quest-collector; no global shell, firewall, or preview files change.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import uuid

PROJECT = Path(__file__).resolve().parents[1]
SDK_VERSION = "6.2.1"
SDK_NAME = f"CloudXR-{SDK_VERSION}-Linux-sdk.tar.gz"
# NVIDIA NGC /versions/6.2.1/files metadata, verified 2026-09-02.
SDK_SHA256 = "3d2f71c07542e7d6225794a8032d8bdfc60072cb6093c62529fe07fc64842c24"
SDK_URL = f"https://api.ngc.nvidia.com/v2/resources/nvidia/cloudxr-runtime/versions/{SDK_VERSION}/files/{SDK_NAME}"
LICENSE_URL = "https://developer.download.nvidia.com/cloudxr/EULA/NVIDIA_CloudXR_GA_License_without_Data_Collection_25Feb2025.pdf"


def run(*args: str | Path, capture: bool = False) -> str:
    result = subprocess.run([str(arg) for arg in args], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def ipv4(value: str) -> str:
    try:
        address = ipaddress.IPv4Address(value)
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError("choose the LAN interface address used by Quest")
        return str(address)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid LAN IPv4: {value}: {exc}") from exc


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract regular files/dirs and same-directory library symlinks only."""
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            target = destination.joinpath(*path.parts)
            if any(parent.is_symlink() for parent in (target, *target.parents)):
                raise ValueError(f"Archive path traverses a symlink: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as incoming, target.open("xb") as output:
                    shutil.copyfileobj(incoming, output)
                target.chmod(member.mode & 0o755)
            elif member.issym():
                link = PurePosixPath(member.linkname)
                if link.is_absolute() or len(link.parts) != 1 or member.linkname in ("", ".", ".."):
                    raise ValueError(f"Unsafe archive symlink: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(member.linkname)
            else:
                raise ValueError(f"Unsupported archive member: {member.name}")


def validate_sdk(directory: Path) -> Path:
    for relative in ("include/cxrServiceAPI.h", "libcloudxr.so", "openxr_cloudxr.json"):
        if not (directory / relative).is_file():
            raise ValueError(f"SDK missing {relative} in {directory}")
    manifest = directory / "openxr_cloudxr.json"
    library = Path(json.loads(manifest.read_text())["runtime"]["library_path"])
    if not (manifest.parent / library).is_file():
        raise ValueError(f"OpenXR manifest library is missing: {library}")
    return manifest


def install_sdk(state: Path, archive: Path | None, download: bool) -> Path:
    installed = state / "runtime" / SDK_VERSION
    if installed.exists():
        validate_sdk(installed)
        print(f"[REUSE] SDK {installed}")
        return installed
    archive = archive or state / "downloads" / SDK_NAME
    if not archive.is_file():
        if not download:
            raise ValueError("SDK missing. Supply --sdk-archive FILE or --download-runtime after reviewing NVIDIA's license.")
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"[DOWNLOAD] NVIDIA CloudXR {SDK_VERSION}; terms: {LICENSE_URL}", flush=True)
        with tempfile.TemporaryDirectory(prefix="download-", dir=archive.parent) as work:
            partial = Path(work) / SDK_NAME
            run("curl", "-fL", "--retry", "2", "--connect-timeout", "15", "--max-time", "180", SDK_URL, "-o", partial)
            verify_checksum(partial)
            partial.rename(archive)
    verify_checksum(archive)
    installed.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extract-", dir=installed.parent) as work:
        staging = Path(work) / "sdk"
        staging.mkdir()
        safe_extract(archive, staging)
        validate_sdk(staging)
        staging.rename(installed)
    print(f"[READY] SDK {installed}")
    return installed


def verify_checksum(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SDK_SHA256:
        raise ValueError(f"SHA-256 mismatch: {path}; expected NVIDIA Runtime {SDK_VERSION}. File preserved for inspection.")


def select_python(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else [
        Path.home() / root / "envs/env_isaaclab_232/bin/python"
        for root in ("anaconda3", "miniconda3", "miniforge3")
    ]
    for path in candidates:
        if path and path.is_file():
            try:
                run(path, "-c", "import sys; from importlib.metadata import version; "
                    "assert sys.version_info[:2] == (3, 11); "
                    "assert version('isaaclab') == '0.54.2'; "
                    "assert version('isaacsim').startswith('5.1.0')", capture=True)
                return path.absolute()
            except subprocess.CalledProcessError:
                continue
    raise ValueError("No supported Isaac Python found. Install per docs/INSTALL.md or pass --isaaclab-python PATH.")


def prepare_certificate(state: Path, host: str, renew: bool) -> tuple[Path, Path]:
    name = host if not renew else f"{host}-{uuid.uuid4().hex[:8]}"
    folder = state / "certs" / name
    cert, key = folder / "server.crt", folder / "server.key"
    if cert.exists() or key.exists():
        if not cert.is_file() or not key.is_file():
            raise ValueError(f"Incomplete certificate pair in {folder}; use --renew-certificate.")
        try:
            run("openssl", "x509", "-in", cert, "-noout", "-checkend", "86400", capture=True)
            run("openssl", "x509", "-in", cert, "-noout", "-checkip", host, capture=True)
            cert_public = run("openssl", "x509", "-in", cert, "-noout", "-pubkey", capture=True)
            key_public = run("openssl", "pkey", "-in", key, "-pubout", capture=True)
            if cert_public != key_public:
                raise ValueError("Certificate and key do not match")
        except (subprocess.CalledProcessError, ValueError) as exc:
            raise ValueError("Certificate expired/invalid. Use --renew-certificate --update-config; old files are retained.") from exc
        key.chmod(0o600)
        return cert, key
    folder.mkdir(parents=True, mode=0o700)
    run("openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-nodes", "-days", "30",
        "-keyout", key, "-out", cert, "-subj", f"/CN={host}",
        "-addext", f"subjectAltName=IP:{host},IP:127.0.0.1,DNS:localhost",
        "-addext", "extendedKeyUsage=serverAuth")
    key.chmod(0o600)
    return cert, key


def snapshot_browser(state: Path, source: Path, refresh: bool) -> Path:
    snapshot = state / "browser"
    if (snapshot / "simple/build/index.html").is_file() and not refresh:
        print(f"[REUSE] Collector web snapshot {snapshot}")
        return snapshot
    if not (source / "index.html").is_file():
        raise ValueError("No built CloudXR web client. See docs/QUEST_COLLECTOR_SETUP.md: prepare the web client, then pass --browser-build DIR.")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Browser build must contain regular files, not symlinks.")
    with tempfile.TemporaryDirectory(prefix="web-", dir=state) as work:
        staging = Path(work) / "browser"
        shutil.copytree(source, staging / "simple/build")
        if snapshot.exists():
            backup = state / f"browser.backup-{uuid.uuid4().hex[:8]}"
            snapshot.rename(backup)
            print(f"[BACKUP] Previous web snapshot: {backup}")
        staging.rename(snapshot)
    return snapshot


def write_env(path: Path, values: dict[str, str], update: bool) -> None:
    content = "# Generated by setup_quest_collector.sh; local paths/secrets must not be committed.\n"
    content += "".join(f"export {key}={shlex.quote(value)}\n" for key, value in values.items())
    if path.exists():
        if path.read_text() == content:
            print(f"[REUSE] Environment {path}")
            return
        if not update:
            raise ValueError(f"{path} differs. Re-run with --update-config to back it up and replace it.")
        backup = path.with_name(f"session.env.backup-{uuid.uuid4().hex[:8]}")
        shutil.copy2(path, backup)
        backup.chmod(0o600)
        print(f"[BACKUP] Environment {backup}")
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        output.write(content)
        pending = Path(output.name)
    pending.chmod(0o600)
    pending.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, type=ipv4, help="PC LAN IPv4 reachable from Quest; never guessed from the default route.")
    parser.add_argument("--web-port", type=int, default=8443, help="Collector HTTPS port, separate from preview's 8080.")
    parser.add_argument("--sdk-archive", type=Path, help=f"Official, unmodified NVIDIA {SDK_NAME}.")
    parser.add_argument("--download-runtime", action="store_true", help=f"Download Runtime {SDK_VERSION} from NVIDIA; review {LICENSE_URL} before using this option.")
    parser.add_argument("--isaaclab-python", type=Path)
    parser.add_argument("--isaaclab-dir", type=Path, default=PROJECT / ".external/IsaacLab-v2.3.2")
    parser.add_argument("--lerobot-python", type=Path, help="Optional separate Python with LeRobot Dataset v3; never installs into Isaac Python.")
    parser.add_argument("--browser-build", type=Path, default=PROJECT / ".external/cloudxr-js-samples/simple/build")
    parser.add_argument("--refresh-web", action="store_true", help="Back up and refresh the collector's web snapshot; never modifies the source build.")
    parser.add_argument("--renew-certificate", action="store_true", help="Generate a new 30-day certificate; preserve old certificate files.")
    parser.add_argument("--update-config", action="store_true", help="Back up and replace differing local environment settings.")
    args = parser.parse_args()
    if not 1024 <= args.web_port <= 65535 or args.web_port in (49100, 8765):
        parser.error("Use a web port in 1024..65535 other than runtime 49100 or preview bridge 8765.")
    if not shutil.which("openssl"):
        parser.error("openssl is required; install it using your OS package manager.")
    state = PROJECT / ".external/quest-collector"
    os.umask(0o077)
    state.mkdir(parents=True, exist_ok=True)
    try:
        python = select_python(args.isaaclab_python)
        lab = args.isaaclab_dir.expanduser().resolve()
        if not (lab / "apps/isaaclab.python.xr.openxr.kit").is_file():
            raise ValueError(f"Isaac Lab OpenXR experience missing under {lab}; pass --isaaclab-dir.")
        sdk = install_sdk(state, args.sdk_archive.expanduser().resolve() if args.sdk_archive else None, args.download_runtime)
        cert, key = prepare_certificate(state, args.host, args.renew_certificate)
        browser = snapshot_browser(state, args.browser_build.expanduser().resolve(), args.refresh_web)
        values = {
            "ISAACLAB_PYTHON": str(python), "ISAACLAB_DIR": str(lab),
            "CLOUDXR_RUNTIME_DIR": str(sdk), "XR_RUNTIME_JSON": str(validate_sdk(sdk)),
            "CLOUDXR_SERVICE_BINARY": str(state / "bin/kuavo-cloudxr-service"),
            "CLOUDXR_HOST": args.host, "QUEST_COLLECTOR_WEB_PORT": str(args.web_port),
            "CLOUDXR_CERTIFICATE": str(cert), "CLOUDXR_KEY": str(key),
            "CLOUDXR_JS_SAMPLES_DIR": str(browser),
        }
        if args.lerobot_python:
            lerobot = args.lerobot_python.expanduser().absolute()
            run(lerobot, "-c", "from lerobot.datasets import CODEBASE_VERSION; assert str(CODEBASE_VERSION) == 'v3.0'", capture=True)
            values["LEROBOT_PYTHON"] = str(lerobot)
        write_env(state / "session.env", values, args.update_config)
        print(f"[READY] {state / 'session.env'}")
        print(f"Quest page: https://{args.host}:{args.web_port}; Manual backend: {args.host}:49100")
        print("Next: ./quest_collector.sh check (no GUI/services), then runtime / web / Quest CONNECT / collect.")
        print("No service was started. No firewall/global shell/preview source settings were changed.")
        print("SDK may bind signaling on all interfaces. Restrict access to your trusted LAN; TLS is not client authentication.")
    except (ValueError, OSError, subprocess.CalledProcessError, tarfile.TarError, KeyError) as exc:
        parser.exit(1, f"[ERROR] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

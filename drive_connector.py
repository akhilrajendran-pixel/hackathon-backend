"""
Google Drive file listing & download, with local fallback.
Supports Shared Drives and recursive subfolder scanning.
"""
import io
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import config

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.presentation",
}

EXPORT_MIME = {
    "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

# Google Docs/Slides get exported to these extensions
GOOGLE_APPS_EXT = {
    "application/vnd.google-apps.document": ".docx",
    "application/vnd.google-apps.presentation": ".pptx",
}

_service = None


def _build_service():
    """Build Google Drive API service from service account credentials."""
    global _service
    if _service is None:
        creds = service_account.Credentials.from_service_account_file(
            config.SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        _service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return _service


def list_drive_files(folder_id: str = None) -> List[Dict]:
    """
    Recursively list all supported files in the Drive folder.
    Supports Shared Drives via supportsAllDrives flag.
    """
    folder_id = folder_id or config.GOOGLE_DRIVE_FOLDER_ID
    service = _build_service()
    all_files = []
    _list_recursive(service, folder_id, all_files)
    logger.info("Found %d supported files in Drive folder %s", len(all_files), folder_id)
    return all_files


def _list_recursive(service, folder_id: str, results: List[Dict]):
    """Recursively scan folder and subfolders."""
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)",
                pageSize=100,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for f in resp.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                # Recurse into subfolder
                _list_recursive(service, f["id"], results)
            elif f["mimeType"] in SUPPORTED_MIME_TYPES:
                # Fix name for Google Docs/Slides (add extension)
                if f["mimeType"] in GOOGLE_APPS_EXT and "." not in f["name"]:
                    f["name"] = f["name"] + GOOGLE_APPS_EXT[f["mimeType"]]
                results.append(f)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def download_drive_file(file_meta: Dict) -> bytes:
    """Download a single file from Drive. Handles native Google Docs export."""
    service = _build_service()
    file_id = file_meta["id"]
    mime = file_meta["mimeType"]

    buf = io.BytesIO()

    if mime in EXPORT_MIME:
        request = service.files().export_media(fileId=file_id, mimeType=EXPORT_MIME[mime])
    else:
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    buf.seek(0)
    return buf.read()


# ── Local fallback connector ────────────────────────────────────────────────

def list_local_files(docs_dir: str = None) -> List[Dict]:
    """List files in the local_docs/ directory with Drive-like metadata shape."""
    docs_dir = docs_dir or config.LOCAL_DOCS_DIR
    results = []
    supported_exts = {".pdf", ".docx", ".pptx"}

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.warning("local_docs directory %s does not exist", docs_dir)
        return results

    for fp in sorted(docs_path.rglob("*")):
        if fp.is_file() and fp.suffix.lower() in supported_exts:
            results.append(
                {
                    "id": fp.name,
                    "name": fp.name,
                    "mimeType": _ext_to_mime(fp.suffix.lower()),
                    "modifiedTime": None,
                    "webViewLink": None,
                    "_local_path": str(fp),
                }
            )

    logger.info("Found %d local files in %s", len(results), docs_dir)
    return results


def download_local_file(file_meta: Dict, docs_dir: str = None) -> bytes:
    """Read a file from the local_docs/ directory."""
    if "_local_path" in file_meta:
        return Path(file_meta["_local_path"]).read_bytes()
    docs_dir = docs_dir or config.LOCAL_DOCS_DIR
    fp = Path(docs_dir) / file_meta["name"]
    return fp.read_bytes()


def _ext_to_mime(ext: str) -> str:
    mime_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    return mime_map.get(ext, "application/octet-stream")


# ── Unified interface ───────────────────────────────────────────────────────

def list_files() -> List[Dict]:
    """Pull from all configured Drive folders; fall back to local_docs/."""
    if config.GOOGLE_DRIVE_FOLDER_IDS and os.path.exists(config.SERVICE_ACCOUNT_FILE):
        all_files = []
        for folder_id in config.GOOGLE_DRIVE_FOLDER_IDS:
            try:
                files = list_drive_files(folder_id)
                all_files.extend(files)
            except Exception as e:
                logger.warning("Drive folder %s not accessible, skipping (%s)", folder_id, e)
        if all_files:
            # Deduplicate by file id
            seen = set()
            unique = []
            for f in all_files:
                if f["id"] not in seen:
                    seen.add(f["id"])
                    unique.append(f)
            logger.info("Total files from %d Drive folder(s): %d", len(config.GOOGLE_DRIVE_FOLDER_IDS), len(unique))
            return unique
        logger.warning("All Drive folders returned 0 files, falling back to local docs")
    return list_local_files()


def download_file(file_meta: Dict) -> bytes:
    """Try Google Drive first; fall back to local_docs/."""
    if "_local_path" in file_meta:
        return download_local_file(file_meta)
    if config.GOOGLE_DRIVE_FOLDER_ID and os.path.exists(config.SERVICE_ACCOUNT_FILE):
        try:
            return download_drive_file(file_meta)
        except Exception as e:
            logger.warning("Drive download failed (%s), falling back to local", e)
    return download_local_file(file_meta)

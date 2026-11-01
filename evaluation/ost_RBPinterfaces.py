#!/usr/bin/env python3
"""
Submit reference (pdb/) and model (af/) CIFs to the SWISS-MODEL OpenStructure Actions API
and store per-interface DockQ and QS-score in pred_protein_rna.

API: https://swissmodel.expasy.org/ost (token auth; compare-structures; poll compare-results)

Structure Assessment (web) vs OST API
  The Assessment server pre-processes uploads (e.g. mmCIF→PDB), which can differ slightly
  from calling the API with raw files—expect edge-case differences. Chain mapping ignores
  chain *names* and uses sequence alignments (see OpenStructure chain mapping docs:
  https://openstructure.org/docs/2.11/mol/alg/chain_mapping ).

Turning **residue-number-alignment** ON (optional)
  By default this script does **not** send ``residue-number-alignment`` (OST aligns by sequence).
  To force it **on** for every submission in this run, use any of:
    ``--residue-number-alignment`` | ``--ost-residue-number-alignment`` | ``--resnum-align``
  Or set env: ``SWISSMODEL_OST_RESNUM_ALIGN=1`` (same effect as the flags).
  Example:
    ``python evaluation/ost_rbp_interfaces.py --residue-number-alignment --overwrite``

OST parameters (defaults aligned with SWISS-MODEL guidance)
  - Do **not** send residue-number-alignment unless you need explicit renumbering; sequence
    alignment is automatic. Use the flags above only when you want that API behaviour.
  - Short nucleotide chains: OST default --min-nuc-length is 4; very short NA may be
    dropped from mapping. With full SEQRES in mmCIF, lowering min-nuc-length is often safe
    (e.g. 3). Set via --min-nuc-length (omit with --no-min-nuc-length to use server default).

Output format in DB (TEXT, JSON), same shape for all interface metrics:
  RBP_DockQ, RBP_QS, RBP_fnat, RBP_irmsd, RBP_lrmsd  ->  [{"A:B": 0.75}, {"A:C": 0.26}, ...]
(RBP_lrmsd = DockQ's ligand RMSD of the interface, the same per-interface list as irmsd/fnat.)
Keys are model chain pairs (mdl_ch1:mdl_ch2). If OST cannot map an interface, lists may
be empty (mdl_chains_without_chem_mapping in JSON).

Uses only the Python standard library for HTTP (urllib); no extra pip packages.

Token: set SWISSMODEL_OST_TOKEN (or pass --token). Create/regenerate at your SWISS-MODEL account:
https://swissmodel.expasy.org/account

Usage (from repo root, after config.json / StartConfig parent is set):
  export SWISSMODEL_OST_TOKEN='...'
  python evaluation/ost_rbp_interfaces.py [--dry-run] [--limit N] [--overwrite]
  python evaluation/ost_rbp_interfaces.py --residue-number-alignment --overwrite   # resnum align ON

Rate limits (server): rapid ~100/min — script sleeps between submissions by default.

Local mode (--local, no token, no network)
  Runs the same OpenStructure action on this machine (``ost compare-structures --dockq --qs-score``
  with the same residue-number-alignment / min-nuc-length options) and feeds its JSON through the
  same parser, so the DB columns are written exactly as from the API. Run it in the 'ost' env:
    conda run -n ost python evaluation/ost_RBPinterfaces.py --local
"""

import argparse
import http.client
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Repo root on path for database.startConfig
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from database.startConfig import StartConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OST_BASE = "https://swissmodel.expasy.org/ost"
DEFAULT_POLL_SEC = 10
DEFAULT_SUBMIT_INTERVAL_SEC = 0.75  # ~80/min, under 100/min rapid limit


def _ensure_text_column(cursor: sqlite3.Cursor, table: str, column: str) -> None:
    try:
        cursor.execute(f'ALTER TABLE {table} ADD COLUMN "{column}" TEXT')
        logger.info('Added column %s.%s (TEXT)', table, column)
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            return
        raise


def _pdb_id_from_af_filename(file_name_cif_af: str) -> str:
    if "_" in file_name_cif_af:
        return file_name_cif_af.split("_")[1].upper()
    return file_name_cif_af[:4].upper()


def _normalize_results_payload(results: Any) -> Optional[Dict[str, Any]]:
    """compare-results may return dict or JSON string."""
    if results is None:
        return None
    if isinstance(results, dict):
        return results
    if isinstance(results, str):
        s = results.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            logger.warning("Could not parse results as JSON (first 200 chars): %s", s[:200])
            return None
    return None


def _chain_mapping_from_ost(data: Dict[str, Any]) -> Dict[str, str]:
    """ref_chain_id -> model_chain_id from compare-structures JSON."""
    cm = data.get("chain_mapping")
    if isinstance(cm, dict):
        return {str(k): str(v) for k, v in cm.items()}
    if isinstance(cm, list):
        out: Dict[str, str] = {}
        for it in cm:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                out[str(it[0])] = str(it[1])
        return out
    return {}


def _unwrap_interface_entry(entry: Any) -> Any:
    """OST sometimes nests rows; API may stringify small lists."""
    if isinstance(entry, str):
        t = entry.strip()
        if t.startswith("["):
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                pass
    return entry


def _interface_model_pair_key(
    entry: Any,
    chain_mapping: Dict[str, str],
    *,
    two_tuple_is_model_pair: bool,
) -> Optional[str]:
    """
    Build 'M1:M2' for DB. OST uses [ref1, ref2, mdl1, mdl2]; reference-only uses [ref1, ref2];
    qs_model_interfaces uses [mdl1, mdl2]. UI also shows 'A⋮B' style strings.
    """
    entry = _unwrap_interface_entry(entry)
    if isinstance(entry, str):
        for sep in ("\u22ee", "⋮", ":"):
            if sep in entry:
                a, b = [p.strip() for p in entry.split(sep, 1)]
                if a and b:
                    return f"{a}:{b}"
        return None
    if isinstance(entry, dict):
        m1 = entry.get("mdl_ch1", entry.get("model_ch1", entry.get("mdl_chain1")))
        m2 = entry.get("mdl_ch2", entry.get("model_ch2", entry.get("mdl_chain2")))
        if m1 is not None and m2 is not None:
            return f"{str(m1).strip()}:{str(m2).strip()}"
        return None
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None
    if len(entry) >= 4:
        m1, m2 = str(entry[2]).strip(), str(entry[3]).strip()
        if m1 and m2:
            return f"{m1}:{m2}"
    if len(entry) == 2:
        a, b = str(entry[0]).strip(), str(entry[1]).strip()
        if not a or not b:
            return None
        if two_tuple_is_model_pair:
            return f"{a}:{b}"
        m1, m2 = chain_mapping.get(a), chain_mapping.get(b)
        if m1 and m2:
            return f"{m1}:{m2}"
        # Reference IDs often match model chain names in AF vs PDB; keep pair for display
        return f"{a}:{b}"
    return None


def _coerce_per_interface_score(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, dict):
        for k in ("dockq", "qs_global", "qs_best", "score", "value"):
            if k in val and val[k] is not None:
                return _coerce_per_interface_score(val[k])
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _as_list_len(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, list):
        return len(x)
    return 0


def _as_list_maybe_json(val: Any) -> Optional[List[Any]]:
    """API may return a real list or a JSON string of a list."""
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        t = val.strip()
        if not t:
            return None
        try:
            parsed = json.loads(t)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def _aligned_iface_score_pairs(interfaces: Any, scores: Any, tag: str) -> List[Tuple[Any, Any]]:
    if not isinstance(interfaces, list) or not interfaces:
        return []
    if not isinstance(scores, list) or not scores:
        return []
    ni, ns = len(interfaces), len(scores)
    n = min(ni, ns)
    if ni != ns:
        logger.warning(
            "%s: interface list length (%d) != score list length (%d); using first %d pairs",
            tag,
            ni,
            ns,
            n,
        )
    return list(zip(interfaces[:n], scores[:n]))


def _pairs_to_db_list(
    pairs: List[Tuple[Any, Any]],
    chain_mapping: Dict[str, str],
    *,
    two_tuple_is_model_pair: bool,
    round_digits: int = 4,
) -> List[Dict[str, float]]:
    # round_digits kept for signature compatibility but intentionally NOT applied: raw full-precision
    # per-interface scores are stored so downstream statistics are not biased by truncation.
    out: List[Dict[str, float]] = []
    for iface, sc in pairs:
        v = _coerce_per_interface_score(sc)
        if v is None:
            continue
        k = _interface_model_pair_key(iface, chain_mapping, two_tuple_is_model_pair=two_tuple_is_model_pair)
        if not k:
            continue
        out.append({k: float(v)})
    return out


def _log_parse_debug(tag: str, interfaces: Any, scores: Any) -> None:
    if not isinstance(interfaces, list) or not interfaces:
        return
    if not isinstance(scores, list) or not scores:
        return
    fi = interfaces[0]
    logger.warning(
        "%s: parser dropped all rows; first interface=%r (type=%s) first score=%r",
        tag,
        fi,
        type(fi).__name__,
        scores[0],
    )


def parse_ost_compare_structures_json(
    data: Dict[str, Any],
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]],
           List[Dict[str, float]]]:
    """
    Per OpenStructure docs (compare-structures JSON):
    - dockq_interfaces + dockq; fnat, irmsd and lrmsd are parallel per-interface lists for the same rows
    - qs_interfaces + per_interface_qs_global
    Fallbacks: dockq_reference_interfaces + dockq (if lengths match); qs_model_interfaces + per_interface_*.
    """
    cmap = _chain_mapping_from_ost(data)
    dock_scores = _as_list_maybe_json(data.get("dockq"))
    if dock_scores is None:
        dock_scores = _as_list_maybe_json(data.get("dockq_scores"))
    qs_g = _as_list_maybe_json(data.get("per_interface_qs_global"))
    qs_b = _as_list_maybe_json(data.get("per_interface_qs_best"))
    fnat_vals = _as_list_maybe_json(data.get("fnat"))
    irmsd_vals = _as_list_maybe_json(data.get("irmsd"))
    if irmsd_vals is None:
        irmsd_vals = _as_list_maybe_json(data.get("iRMSD"))
    lrmsd_vals = _as_list_maybe_json(data.get("lrmsd"))
    if lrmsd_vals is None:
        lrmsd_vals = _as_list_maybe_json(data.get("LRMSD"))

    dock_list: List[Dict[str, float]] = []
    dock_ifaces_used: Optional[List[Any]] = None
    dock_components_tag = ""

    for iface_key, tag in (
        ("dockq_interfaces", "dockq_interfaces+dockq"),
        ("dockq_reference_interfaces", "dockq_reference_interfaces+dockq"),
    ):
        ifaces = _as_list_maybe_json(data.get(iface_key))
        if not ifaces or not dock_scores:
            continue
        pairs = _aligned_iface_score_pairs(ifaces, dock_scores, tag)
        dock_list = _pairs_to_db_list(pairs, cmap, two_tuple_is_model_pair=False)
        dock_ifaces_used = ifaces
        dock_components_tag = tag
        if not dock_list and pairs:
            _log_parse_debug(tag, ifaces, dock_scores)
        break

    fnat_list: List[Dict[str, float]] = []
    irmsd_list: List[Dict[str, float]] = []
    lrmsd_list: List[Dict[str, float]] = []
    if dock_ifaces_used is not None:
        if fnat_vals:
            pairs = _aligned_iface_score_pairs(
                dock_ifaces_used, fnat_vals, f"{dock_components_tag}+fnat"
            )
            fnat_list = _pairs_to_db_list(
                pairs, cmap, two_tuple_is_model_pair=False, round_digits=4
            )
            if not fnat_list and pairs:
                _log_parse_debug(f"{dock_components_tag}+fnat", dock_ifaces_used, fnat_vals)
        if irmsd_vals:
            pairs = _aligned_iface_score_pairs(
                dock_ifaces_used, irmsd_vals, f"{dock_components_tag}+irmsd"
            )
            irmsd_list = _pairs_to_db_list(
                pairs, cmap, two_tuple_is_model_pair=False, round_digits=5
            )
            if not irmsd_list and pairs:
                _log_parse_debug(f"{dock_components_tag}+irmsd", dock_ifaces_used, irmsd_vals)
        if lrmsd_vals:
            pairs = _aligned_iface_score_pairs(
                dock_ifaces_used, lrmsd_vals, f"{dock_components_tag}+lrmsd"
            )
            lrmsd_list = _pairs_to_db_list(
                pairs, cmap, two_tuple_is_model_pair=False, round_digits=5
            )
            if not lrmsd_list and pairs:
                _log_parse_debug(f"{dock_components_tag}+lrmsd", dock_ifaces_used, lrmsd_vals)

    # If dockq was empty but DockQ components exist per reference interface
    ref_if = _as_list_maybe_json(data.get("dockq_reference_interfaces"))
    if ref_if:
        if not fnat_list and fnat_vals:
            pairs = _aligned_iface_score_pairs(ref_if, fnat_vals, "dockq_reference_interfaces+fnat")
            fnat_list = _pairs_to_db_list(
                pairs, cmap, two_tuple_is_model_pair=False, round_digits=4
            )
        if not irmsd_list and irmsd_vals:
            pairs = _aligned_iface_score_pairs(ref_if, irmsd_vals, "dockq_reference_interfaces+irmsd")
            irmsd_list = _pairs_to_db_list(
                pairs, cmap, two_tuple_is_model_pair=False, round_digits=5
            )
        if not lrmsd_list and lrmsd_vals:
            pairs = _aligned_iface_score_pairs(ref_if, lrmsd_vals, "dockq_reference_interfaces+lrmsd")
            lrmsd_list = _pairs_to_db_list(
                pairs, cmap, two_tuple_is_model_pair=False, round_digits=5
            )

    qs_list: List[Dict[str, float]] = []

    qs_if = _as_list_maybe_json(data.get("qs_interfaces"))
    if qs_if and qs_g:
        pairs = _aligned_iface_score_pairs(qs_if, qs_g, "qs_interfaces+per_interface_qs_global")
        qs_list = _pairs_to_db_list(pairs, cmap, two_tuple_is_model_pair=False)
        if not qs_list and pairs:
            _log_parse_debug("qs_interfaces+per_interface_qs_global", qs_if, qs_g)

    qs_mdl_if = _as_list_maybe_json(data.get("qs_model_interfaces"))
    if not qs_list and qs_mdl_if and qs_g:
        pairs = _aligned_iface_score_pairs(qs_mdl_if, qs_g, "qs_model_interfaces+per_interface_qs_global")
        qs_list = _pairs_to_db_list(pairs, cmap, two_tuple_is_model_pair=True)
        if not qs_list and pairs:
            _log_parse_debug("qs_model_interfaces+per_interface_qs_global", qs_mdl_if, qs_g)

    if not qs_list and qs_mdl_if and qs_b:
        pairs = _aligned_iface_score_pairs(qs_mdl_if, qs_b, "qs_model_interfaces+per_interface_qs_best")
        qs_list = _pairs_to_db_list(pairs, cmap, two_tuple_is_model_pair=True)

    dock_if = _as_list_maybe_json(data.get("dockq_interfaces"))
    if not qs_list and dock_if and qs_g:
        pairs = _aligned_iface_score_pairs(dock_if, qs_g, "dockq_interfaces+per_interface_qs_global")
        qs_list = _pairs_to_db_list(pairs, cmap, two_tuple_is_model_pair=False)

    if not dock_list and not qs_list and not fnat_list and not irmsd_list:
        logger.warning(
            "No per-interface DockQ/QS/fnat/iRMSD rows extracted. Lengths: dockq_if=%d dockq_ref_if=%d "
            "dockq_scores=%d fnat=%d irmsd=%d qs_if=%d qs_model_if=%d per_interface_qs_global=%d "
            "mdl_chains_without_chem_mapping=%s",
            _as_list_len(data.get("dockq_interfaces")),
            _as_list_len(data.get("dockq_reference_interfaces")),
            _as_list_len(dock_scores),
            _as_list_len(fnat_vals),
            _as_list_len(irmsd_vals),
            _as_list_len(data.get("qs_interfaces")),
            _as_list_len(data.get("qs_model_interfaces")),
            _as_list_len(qs_g),
            data.get("mdl_chains_without_chem_mapping"),
        )

    return dock_list, qs_list, fnat_list, irmsd_list, lrmsd_list


def _auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Token {token.strip()}"}


def _build_multipart_form_data(
    fields: Dict[str, str],
    files: List[Tuple[str, str, bytes]],
) -> Tuple[bytes, str]:
    """Return (body_bytes, content_type) for multipart/form-data."""
    boundary = f"------------------------{uuid.uuid4().hex}"
    crlf = b"\r\n"
    parts: List[bytes] = []
    bnd = boundary.encode("ascii")

    for name, value in fields.items():
        parts.append(b"--" + bnd + crlf)
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode("ascii") + crlf + crlf)
        parts.append(value.encode("utf-8") + crlf)

    for field_name, filename, content in files:
        parts.append(b"--" + bnd + crlf)
        disp = (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode("ascii")
            + crlf
            + b"Content-Type: application/octet-stream"
            + crlf
            + crlf
        )
        parts.append(disp)
        parts.append(content + crlf)

    parts.append(b"--" + bnd + b"--" + crlf)
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


SUBMIT_ATTEMPTS = 4  # POST compare-structures tries per structure before giving up on it


def _http_request(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[bytes] = None,
    timeout: float = 600,
) -> Tuple[int, Dict[str, str], str]:
    """Return (status_code, headers_lower_keys_from_get, body_text)."""
    hdrs = dict(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            h = {k: v for k, v in resp.headers.items()}
            text = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), h, text
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        h = {k: v for k, v in (e.headers.items() if e.headers else [])}
        return e.code, h, text
    except urllib.error.URLError as e:
        raise RuntimeError(str(e.reason) if getattr(e, "reason", None) else str(e)) from e
    except (TimeoutError, ConnectionError, http.client.HTTPException, OSError) as e:
        # Raised while reading the response (read timeout, SSL EOF, reset/broken pipe); not wrapped
        # in URLError by urllib. Report like any other transient network failure instead of crashing.
        raise RuntimeError(f"{type(e).__name__}: {e}") from e


def submit_compare_structures(
    token: str,
    reference_path: str,
    model_path: str,
    *,
    residue_number_alignment: bool = False,
    min_nuc_length: Optional[int] = 3,
) -> Tuple[Optional[str], Optional[str]]:
    """
    POST compare-structures. Returns (project_id, error_message).

    By default matches SWISS-MODEL's suggestion: leave residue-number-alignment off unless
    needed; optional min-nuc-length for short nucleotide chains in chain mapping.
    """
    url = f"{OST_BASE}/compare-structures"
    # Form flags (strings are safest for servers expecting text fields).
    form_fields = {
        "dockq": "true",
        "qs-score": "true",
    }
    if residue_number_alignment:
        form_fields["residue-number-alignment"] = "true"
    if min_nuc_length is not None:
        form_fields["min-nuc-length"] = str(int(min_nuc_length))
    try:
        with open(reference_path, "rb") as ref_b, open(model_path, "rb") as mdl_b:
            ref_bytes = ref_b.read()
            mdl_bytes = mdl_b.read()
    except OSError as e:
        return None, str(e)

    body, ctype = _build_multipart_form_data(
        form_fields,
        [
            ("reference", os.path.basename(reference_path), ref_bytes),
            ("model", os.path.basename(model_path), mdl_bytes),
        ],
    )
    headers = _auth_headers(token)
    headers["Content-Type"] = ctype

    # Retry transient network failures (SSL EOF, handshake/read timeouts, broken pipe) with backoff.
    for attempt in range(1, SUBMIT_ATTEMPTS + 1):
        try:
            status, _resp_h, text = _http_request("POST", url, headers=headers, data=body, timeout=600)
            break
        except RuntimeError as e:
            if attempt == SUBMIT_ATTEMPTS:
                return None, f"{e} (after {attempt} attempts)"
            wait = 30 * attempt
            logger.warning("submit attempt %d/%d failed (%s); retrying in %ds", attempt, SUBMIT_ATTEMPTS, e, wait)
            time.sleep(wait)

    if status == 429:
        return None, f"429 rate limited: {text[:500]}"
    if status >= 400:
        return None, f"HTTP {status}: {text[:800]}"

    try:
        body_json = json.loads(text)
    except json.JSONDecodeError:
        return None, f"Non-JSON response: {text[:500]}"

    pid = body_json.get("project_id")
    if not pid:
        return None, f"No project_id in response: {body_json}"
    return str(pid), None


def poll_project_until_done(
    token: str,
    project_id: str,
    poll_sec: float = DEFAULT_POLL_SEC,
    max_wait_sec: float = 7200,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    GET compare-results until COMPLETED or FAILED. Returns (results_dict_or_none, status).
    """
    base = f"{OST_BASE}/compare-results/"
    qs = urllib.parse.urlencode({"project_id": project_id})
    url = f"{base}?{qs}"
    deadline = time.time() + max_wait_sec
    last_status = "UNKNOWN"

    while time.time() < deadline:
        time.sleep(poll_sec)
        try:
            status, resp_h, text = _http_request(
                "GET", url, headers=_auth_headers(token), timeout=120
            )
        except RuntimeError as e:
            logger.warning("poll request failed: %s", e)
            continue

        if status == 429:
            ra = resp_h.get("Retry-After") or resp_h.get("retry-after")
            wait = float(ra) if ra and str(ra).isdigit() else 60.0
            logger.warning("429 while polling; sleeping %.1fs", wait)
            time.sleep(wait)
            continue

        if status >= 400:
            logger.warning("compare-results HTTP %s: %s", status, text[:400])
            continue

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue

        projects = payload.get("projects") or []
        if not projects:
            continue
        proj = projects[0]
        last_status = str(proj.get("status", "UNKNOWN"))
        logger.info("project_id=%s status=%s", project_id, last_status)

        if last_status == "COMPLETED":
            raw = proj.get("results")
            data = _normalize_results_payload(raw)
            return data, last_status
        if last_status == "FAILED":
            return None, last_status

    return None, f"TIMEOUT(last={last_status})"


def run_local_compare_structures(
    reference_path: str,
    model_path: str,
    *,
    residue_number_alignment: bool = False,
    min_nuc_length: Optional[int] = 3,
    max_wait_sec: float = 7200,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Local counterpart of submit_compare_structures + poll_project_until_done: run
    ``ost compare-structures`` with the same options and return (results_dict_or_none, status).
    """
    import subprocess
    import tempfile

    ost_bin = os.path.join(sys.prefix, "bin", "ost")
    if not os.path.isfile(ost_bin):
        ost_bin = "ost"
    with tempfile.TemporaryDirectory(prefix="ost_rbp_") as tmp:
        out_json = os.path.join(tmp, "out.json")
        cmd = [ost_bin, "compare-structures", "-m", model_path, "-r", reference_path,
               "-o", out_json, "--dockq", "--qs-score"]
        if residue_number_alignment:
            cmd.append("--residue-number-alignment")
        if min_nuc_length is not None:
            cmd += ["--min-nuc-length", str(int(min_nuc_length))]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max_wait_sec)
        except subprocess.TimeoutExpired:
            return None, f"TIMEOUT({max_wait_sec:.0f}s)"
        if not os.path.isfile(out_json):
            logger.warning("ost compare-structures rc=%s: %s", proc.returncode, proc.stderr[-800:])
            return None, "FAILED"
        with open(out_json) as fh:
            data = json.load(fh)
    if data.get("status") == "FAILURE":
        return None, "FAILED"
    return data, "COMPLETED"


def process_rows(
    db_path: str,
    table_name: str,
    parent_folder: str,
    token: str,
    dry_run: bool,
    limit: Optional[int],
    overwrite: bool,
    submit_interval: float,
    poll_sec: float,
    residue_number_alignment: bool,
    min_nuc_length: Optional[int],
    local: bool = False,
) -> None:
    af_folder = os.path.join(parent_folder, "af")
    pdb_folder = os.path.join(parent_folder, "pdb")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    _ensure_text_column(cursor, table_name, "RBP_DockQ")
    _ensure_text_column(cursor, table_name, "RBP_QS")
    _ensure_text_column(cursor, table_name, "RBP_fnat")
    _ensure_text_column(cursor, table_name, "RBP_irmsd")
    _ensure_text_column(cursor, table_name, "RBP_lrmsd")
    if not dry_run:
        conn.commit()

    q = f'SELECT Id, FileName, "RBP_DockQ", "RBP_QS", "RBP_fnat", "RBP_irmsd" FROM {table_name} WHERE FileName IS NOT NULL AND FileName != ""'
    if not overwrite:
        q += ' AND ("RBP_DockQ" IS NULL OR "RBP_DockQ" = "")'
    q += " ORDER BY Id"
    if limit is not None:
        q += " LIMIT ?"
        cursor.execute(q, (int(limit),))
    else:
        cursor.execute(q)
    rows = cursor.fetchall()
    logger.info("Rows to process: %d (table=%s)", len(rows), table_name)

    processed = 0
    from tqdm import tqdm
    for row in tqdm(rows, total=len(rows), desc="SWISS-MODEL RBP interfaces", unit="struct"):
        row_id = row["Id"]
        file_name = row["FileName"]
        pdb_id = _pdb_id_from_af_filename(file_name)
        ref_path = os.path.join(pdb_folder, f"{pdb_id}.cif")
        if not os.path.isfile(ref_path):  # case-sensitive file systems: pdb/ files are usually lower case
            ref_path = os.path.join(pdb_folder, f"{pdb_id.lower()}.cif")
        model_path = os.path.join(af_folder, file_name)

        if not os.path.isfile(ref_path):
            logger.warning("Skip id=%s: missing reference %s", row_id, ref_path)
            continue
        if not os.path.isfile(model_path):
            logger.warning("Skip id=%s: missing model %s", row_id, model_path)
            continue

        logger.info(
            "Submitting id=%s pdb=%s model=%s (residue_number_alignment=%s min_nuc_length=%s)",
            row_id,
            pdb_id,
            file_name,
            residue_number_alignment,
            min_nuc_length,
        )
        if dry_run:
            logger.info("[DRY] would POST compare-structures and update RBP_DockQ / RBP_QS / RBP_fnat / RBP_irmsd / RBP_lrmsd")
            processed += 1
            continue

        if local:
            pid = "local"
            results, status = run_local_compare_structures(
                ref_path,
                model_path,
                residue_number_alignment=residue_number_alignment,
                min_nuc_length=min_nuc_length,
            )
        else:
            pid, err = submit_compare_structures(
                token,
                ref_path,
                model_path,
                residue_number_alignment=residue_number_alignment,
                min_nuc_length=min_nuc_length,
            )
            if err:
                logger.error("Submit failed id=%s: %s", row_id, err)
                continue

            results, status = poll_project_until_done(token, pid, poll_sec=poll_sec)
        if status != "COMPLETED" or not results:
            logger.error("Job id=%s project=%s finished with status=%s (no results)", row_id, pid, status)
            continue

        dock_list, qs_list, fnat_list, irmsd_list, lrmsd_list = parse_ost_compare_structures_json(results)
        if not dock_list and not qs_list and not fnat_list and not irmsd_list:
            logger.warning(
                "id=%s: parsed empty interface lists; check OST JSON keys. Sample keys: %s",
                row_id,
                list(results.keys())[:30],
            )

        dock_json = json.dumps(dock_list)
        qs_json = json.dumps(qs_list)
        fnat_json = json.dumps(fnat_list)
        irmsd_json = json.dumps(irmsd_list)
        lrmsd_json = json.dumps(lrmsd_list)
        cursor.execute(
            f'UPDATE {table_name} SET "RBP_DockQ" = ?, "RBP_QS" = ?, "RBP_fnat" = ?, "RBP_irmsd" = ?, '
            f'"RBP_lrmsd" = ? WHERE Id = ?',
            (dock_json, qs_json, fnat_json, irmsd_json, lrmsd_json, row_id),
        )
        conn.commit()
        processed += 1
        logger.info(
            "Updated id=%s: DockQ=%d QS=%d fnat=%d iRMSD=%d LRMSD=%d interface pairs",
            row_id,
            len(dock_list),
            len(qs_list),
            len(fnat_list),
            len(irmsd_list),
            len(lrmsd_list),
        )

        if submit_interval > 0 and not local:
            time.sleep(submit_interval)

    conn.close()
    logger.info("Done. updated_rows=%d dry_run=%s", processed, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWISS-MODEL OST compare-structures -> RBP_DockQ, RBP_QS, RBP_fnat, RBP_irmsd, RBP_lrmsd"
    )
    parser.add_argument("--token", type=str, default=None, help="API token (else env SWISSMODEL_OST_TOKEN)")
    parser.add_argument("--table", type=str, default=None, help="DB table (default: StartConfig pred table)")
    parser.add_argument("--dry-run", action="store_true", help="List work only; no HTTP or DB writes")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite DB: re-run API and replace RBP_* interface columns even when already set",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process")
    parser.add_argument("--submit-interval", type=float, default=DEFAULT_SUBMIT_INTERVAL_SEC, help="Seconds between submissions")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_SEC, help="Seconds between status polls")
    parser.add_argument(
        "--residue-number-alignment",
        "--ost-residue-number-alignment",
        "--resnum-align",
        action="store_true",
        dest="residue_number_alignment",
        help=(
            "Send residue-number-alignment=true on every compare-structures POST (OST uses residue "
            "numbers for chem-group alignments). Default is OFF. Same as env SWISSMODEL_OST_RESNUM_ALIGN=1."
        ),
    )
    parser.add_argument(
        "--min-nuc-length",
        type=int,
        default=3,
        metavar="N",
        help="OST min-nuc-length for chain mapping (OST default is 4; 3 helps short NA). Ignored if --no-min-nuc-length.",
    )
    parser.add_argument(
        "--no-min-nuc-length",
        action="store_true",
        help="Do not send min-nuc-length (use server/OpenStructure default, typically 4)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run 'ost compare-structures' locally instead of the SWISS-MODEL API (no token needed)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("SWISSMODEL_OST_TOKEN") or os.environ.get("SWISSMODEL_API_TOKEN")
    if not args.dry_run and not args.local and not token:
        raise SystemExit("Set SWISSMODEL_OST_TOKEN or pass --token")

    cfg = StartConfig()
    parent = cfg.get_parent_folder()
    if not parent:
        raise SystemExit("StartConfig parent_folder not set. Run database/startConfig.py with your data folder or fix config.json")
    db_path = cfg.get_database_path()
    table = args.table or cfg.get_predicted_table_name()

    resnum_align = bool(args.residue_number_alignment)
    if not resnum_align:
        resnum_align = os.environ.get("SWISSMODEL_OST_RESNUM_ALIGN", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    if resnum_align:
        logger.info("residue-number-alignment is ON (API field residue-number-alignment=true)")

    process_rows(
        db_path=db_path,
        table_name=table,
        parent_folder=parent,
        token=token or "",
        dry_run=args.dry_run,
        limit=args.limit,
        overwrite=args.overwrite or args.force,
        submit_interval=args.submit_interval,
        poll_sec=args.poll_interval,
        residue_number_alignment=resnum_align,
        min_nuc_length=None if args.no_min_nuc_length else args.min_nuc_length,
        local=args.local,
    )


if __name__ == "__main__":
    # Quiet per-sample INFO chatter so the tqdm progress bar renders cleanly (warnings/errors stay).
    logging.getLogger().setLevel(logging.WARNING)
    main()

# CodeQL Security Scan — msocr

> CodeQL 2.25.6, Python extractor, `codeql/python-queries@1.8.4`. Database
> built from the repo root (`--source-root=.`), 56 Python files scanned, 45
> queries evaluated. SARIF artifact: `codeql-results.sarif` (repo root, gitignored).

## Run

```bash
codeql database create .codeql-db --language=python --source-root=. --overwrite
codeql database analyze .codeql-db codeql/python-queries \
  --format=sarif-latest --output=codeql-results.sarif --threads=0
```

## Summary

| Rule | Findings | Verified severity |
|---|---|---|
| `py/path-injection` (CWE-022) | 11 | **3 high** (HTTP API), 8 low (internal CLI) |
| `py/paramiko-missing-host-key-validation` (CWE-295) | 3 | **medium** (RunPod SSH) |
| `py/xml-bomb` (CWE-776) | 1 | **medium** (PAGE XML parser) |

Total: 15 raw findings. After triage against source: **7 actionable**.

---

## HIGH — Path injection in HTTP API (3 findings)

All three are in `msocr/service/annotation_api.py` and take a path from an HTTP
request body/query/path-param and pass it to `Path()` / `_C2AV_PLATES / filename`
without confining it to a safe root.

### H1 — `annotation_api.py:536` — `POST /api/sessions/{id}/import-xml`

```python
# body/query: {"xml_path": "/abs/path/to.xml"} or ?xml_path=...
p = Path(xml_path)
if not p.is_absolute():
    p = Path.cwd() / p
if not p.exists():                       # ← flagged: user-controlled path
    raise HTTPException(404, ...)
updated = manager.import_v2_from_xml(session_id, p)   # parses arbitrary XML
```

**Risk:** an attacker who can reach the annotation API (port 8001) can make the
server read and parse *any* file the process user can read, including
`/etc/passwd`-class targets and, via the PAGE-XML parser (see M3), arbitrary
XML that may expand entities. The `xml_path` value is fully attacker-controlled.

**Fix:** require an absolute path under a configured corpus root, reject `..`,
and validate the resolved path stays inside the root:
```python
CORPUS_ROOT = Path(os.environ.get("MSOCR_CORPUS_ROOT", ".")).resolve()
p = (CORPUS_ROOT / xml_path).resolve() if not Path(xml_path).is_absolute() else Path(xml_path).resolve()
if CORPUS_ROOT not in p.parents and p != CORPUS_ROOT:
    raise HTTPException(403, "path outside corpus root")
```

### H2/H3 — `annotation_api.py:617, 637` — C2AV plate endpoints

```python
# GET /api/corpus/c2av/plates/{filename}/thumbnail
plate_path = _C2AV_PLATES / filename
if not plate_path.exists() or not filename.endswith(".png"):   # ← flagged
    ...
with Image.open(plate_path) as img: ...
```

and the symmetric `POST .../plates/{filename}/session` at line 637.

**Risk:** `{filename}` is a path parameter. `_C2AV_PLATES / filename` with
`filename = "../secret.png"` resolves outside the plates dir. The `.endswith(".png")`
check does not stop traversal. `Image.open` will happily read any PNG on disk,
and the response is returned as JPEG bytes — a limited but real exfiltration
channel. The session-creation variant also feeds the path into a new annotation
session as the image source.

**Fix:** reject anything that escapes the plates directory:
```python
plate_path = (_C2AV_PLATES / filename).resolve()
if plate_path.parent != _C2AV_PLATES.resolve() or not filename.endswith(".png"):
    raise HTTPException(404, "Plate not found")
```

### Note on the SPA catch-all (`annotation_api.py:668, 670`)

Two `py/path-injection` hits at lines 668 and 670 are on the SPA catch-all:
```python
candidate = _FRONTEND_DIST / full_path
if full_path and candidate.is_file():                          # ← 668
    media_type, _ = mimetypes.guess_type(str(candidate))
    return FileResponse(candidate, media_type=media_type or "application/octet-stream")
```
`_FRONTEND_DIST / full_path` with `full_path = "../settings.py"` can escape the
dist dir. **Severity: medium-high** — it returns arbitrary small files from the
repo. Same fix as H2: confine `candidate.resolve().parent` to `_FRONTEND_DIST.resolve()`.
Counted here, not separately, because it is one code path.

---

## MEDIUM — Paramiko AutoAddPolicy (3 findings)

`msocr/training/runpod_runner.py:77, 103, 119` — three identical:
```python
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
```

**Risk:** the RunPod runner connects to freshly-spawned GPU pods over SSH and
accepts any host key on first connect. For ephemeral pods whose host identity
is not pinned, this is the documented MITM risk of `AutoAddPolicy`: a DNS/ARP
poisoner between you and the pod can impersonate the pod and capture the SSH
key. In a trusted RunPod-network context the practical risk is low, but the
correct pattern for ephemeral infra is to fetch the pod's reported SSH
fingerprint from the RunPod API and compare, rather than blindly accepting.

**Fix (lazy, correct for ephemeral pods):** pin the host key from RunPod's
`pod.hostShipport` / SSH endpoint data via a `MissingHostKeyPolicy` that
verifies against an expected key, falling back to `RejectPolicy` otherwise.
At minimum, set a known-hosts file per run and use it. This is a deliberate
simplification with a known ceiling — add a `# ponytail:` comment naming it.

---

## MEDIUM — XML bomb / entity expansion (1 finding)

`msocr/data/session_manager.py:466` — `import_v2_from_xml`:
```python
tree = ET.parse(xml_path)    # ← flagged: user-provided XML, no defusedxml
root = tree.getroot()
```

**Risk:** the standard-library `xml.etree.ElementTree` is vulnerable to
internal entity expansion ("billion laughs") and external entity injection
(XXE) per the Python docs security note. The XML here comes from the annotation
API's `import-xml` endpoint (H1), so the input is attacker-controlled in the
threat model where the API is exposed. Even without H1, any future code path
that feeds user XML here inherits the risk.

**Fix:** use `defusedxml.ElementTree` (drop-in):
```python
from defusedxml.ElementTree import parse
tree = parse(xml_path)
```
Add `defusedxml` to dependencies. `defusedxml` is the Python-recommended
hardened parser and disables entity expansion by default.

---

## LOW — Internal CLI/runtime path checks (8 findings, false-positive-ish)

These are flagged because CodeQL tracks `Path(model_path).exists()` where
`model_path` originates from CLI args or env vars, but the threat model is
"local user running the CLI on their own machine" — not a remote attacker.

| File:line | Code | Real risk |
|---|---|---|
| `models/inference.py:30` | `self.model_path = Path(model_path); if not self.model_path.exists()` | None — CLI/local |
| `utils/input_loader.py:23` | `if not input_path.exists()` | None — CLI/local |
| `service/runtime.py:93` | `model_path = Path(resolved_model); if not model_path.exists()` | None — env-var/local |
| `data/session_manager.py:461` | `if not xml_path.exists()` | None — called from internal paths |
| `data/session_manager.py:466` | `tree = ET.parse(xml_path)` | M3 above (the parse itself) |
| `data/session_manager.py:653` | (session image path) | None — internal |

These do not need code changes; they are the normal shape of a CLI tool that
reads files named by the operator. Listed for completeness so the SARIF noise is
explained.

---

## Actionable fix list (priority order)

1. **H1** — confine `import-xml` `xml_path` to a corpus root; reject traversal. (annotation_api.py:536)
2. **H2/H3 + SPA** — confine C2AV `filename` and SPA `full_path` to their dist roots; reject `..`. (annotation_api.py:617, 637, 668)
3. **M3** — swap `xml.etree.ElementTree` → `defusedxml.ElementTree` in `session_manager.py:466` and anywhere else PAGE XML is parsed. Add `defusedxml` to `pyproject.toml`.
4. **M1/M2** — replace `paramiko.AutoAddPolicy()` with a pinned-host-key policy in `runpod_runner.py:77, 103, 119`. Document the expected-key source (RunPod API).

Items 1–3 are bounded, single-file fixes and are good candidates for a single
`@fixer` pass once you confirm the corpus-root and dist-root policy. Item 4
needs a small design decision on where to store expected SSH keys (env var?
RunPod API call?).

## Coverage notes

- CodeQL scanned 56 Python files; the repo has ~16 modules under `msocr/`.
  The extra count includes test files and generated stubs in the source root.
- Only the `codeql/python-queries` pack ran. The `python-security-extended`
  pack was not available without registry auth — re-run with that pack once
  credentials are configured for deeper coverage (taint-style queries, more
  CWEs).
- No findings for: command injection, SSRF, SQL/LDAP/XPath injection, ReDoS,
  weak crypto, insecure cookie/temp-file, cleartext logging, missing TLS
  host-key (other than paramiko), reflected XSS, CSRF. These query classes
  ran and returned zero results.
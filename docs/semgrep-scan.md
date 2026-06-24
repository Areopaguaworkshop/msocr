# Semgrep Security Scan — msocr

> Semgrep 1.165.0. Two rule sets run: `--config auto` (504 rules) and
> `p/owasp-top-ten` (251 rules). 1770 targets scanned, ~99.9% parsed.
> Artifacts: `semgrep-auto.json`, `semgrep-owasp.json` (repo root, gitignored).

## Run

```bash
semgrep scan --config auto . --json --output semgrep-auto.json
semgrep scan --config p/owasp-top-ten . --json --output semgrep-owasp.json
```

## Summary

| Rule | Sev | Findings | Verified |
|---|---|---|---|
| `python.lang.security.use-defused-xml-parse` | ERROR | 2 | **2 real** — XXE/billion-laughs |
| `python.lang.security.audit.dynamic-urllib-use-detected` | WARNING | 5 | **1 real** (IIIF SSRF), 4 false-positive (health checks) |

Total: 7 raw → **3 actionable** after triage.

Cross-check: the two XXE findings overlap with `docs/codeql-scan.md` finding M3
(`session_manager.py:466`). Semgrep additionally caught the second parser site
in `utils/__init__.py:134` that CodeQL's `py/xml-bomb` did not flag.

---

## ERROR — XXE / billion-laughs (2 findings, both real)

### E1 — `session_manager.py:466`

```python
def _fetch_iiif_image(self, manifest_url: str) -> Tuple[bytes, str]:
    ...
# (in import_v2_from_xml)
import xml.etree.ElementTree as ET
tree = ET.parse(xml_path)    # ← ERROR: stdlib ET is XXE-vulnerable
```

Same site flagged by CodeQL (`docs/codeql-scan.md` M3). The XML input reaches
here from the `POST /api/sessions/{id}/import-xml` endpoint (CodeQL H1), so
the input is attacker-controlled in the exposed-API threat model.

**Fix:** swap to `defusedxml.ElementTree` and add `defusedxml` to deps:
```python
from defusedxml.ElementTree import parse
tree = parse(xml_path)
```

### E2 — `utils/__init__.py:134`

```python
def validate_alto_or_page_xml(xml_path) -> dict:
    """Validate ALTO or PAGE XML file and return metadata."""
    ...
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)    # ← ERROR
```

A second PAGE/ALTO parser in `msocr/utils/__init__.py` that CodeQL did not
catch. This is a validation helper; its caller paths need auditing, but the
fix is identical regardless of caller.

**Fix:** same `defusedxml` swap.

Both E1 and E2 are fixed by one dependency change and two import swaps. They
are the highest-value single fix in either scan.

---

## WARNING — dynamic urllib use / SSRF (5 findings, 1 real)

### W1 — `session_manager.py:666, 670` — IIIF fetch (REAL)

```python
def _fetch_iiif_image(self, manifest_url: str) -> Tuple[bytes, str]:
    with urlopen(manifest_url) as response:          # ← 666
        manifest = json.loads(response.read().decode("utf-8"))
    image_url = self._extract_iiif_image_url(manifest)
    with urlopen(image_url) as response:              # ← 670
        image_bytes = response.read()
```

`manifest_url` originates from session creation (`ingestion_path=iiif`), and
`image_url` is extracted from the remote manifest — fully attacker-controlled
if the IIIF manifest URL is user-supplied. `urllib.urlopen` accepts `file://`,
so this is an SSRF + local-file-read surface: a malicious manifest URL of
`file:///etc/passwd` would return the file's bytes, and a malicious manifest
can point `image_url` anywhere reachable by the process.

**Fix (lazy, correct):** enforce an `http(s)://` scheme and an optional
allowlist of IIIF hosts before `urlopen`:
```python
from urllib.parse import urlparse
def _safe_urlopen(url: str, *a, **kw):
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) URL: {url!r}")
    return urlopen(url, *a, **kw)
```
Then use `_safe_urlopen` at lines 666 and 670. If IIIF sources are a fixed set,
add a host allowlist.

### W2–W5 — `deploy.py:95, 189, 234` — FALSE POSITIVES

```python
# 95: health check against a built-from-config health_url
req = request.Request(health_url, method="GET")
with request.urlopen(req, timeout=min(timeout_sec, 10)) as response:
# 189: runtime HTR smoke test, req built from content_type + payload above
with request.urlopen(req, timeout=timeout_sec) as response:
# 234: Gradio readiness check against root_url
with request.urlopen(req, timeout=min(timeout_sec, 10)) as response:
```

All three are server-readiness loops where the URL is built from
deploy/config constants (`health_url`, `root_url`) or from an internal smoke
test payload — not from untrusted request input. Semgrep flagged them because
the URL variable is not a literal, but the data flow is config → URL, not
request → URL. No fix needed. Listed for completeness.

---

## Cross-scan reconciliation (Semgrep × CodeQL)

| Issue | CodeQL | Semgrep | Notes |
|---|---|---|---|
| XXE `session_manager.py:466` | `py/xml-bomb` ✓ | `use-defused-xml-parse` ✓ | Both caught. Fix once. |
| XXE `utils/__init__.py:134` | — | `use-defused-xml-parse` ✓ | Semgrep-only. Fix in same pass. |
| IIIF SSRF `session_manager.py:666/670` | — | `dynamic-urllib-use-detected` ✓ | Semgrep-only. Related to CodeQL H1 path flow but distinct vector. |
| Path injection (annotation_api.py) | `py/path-injection` ×5 ✓ | — | CodeQL-only. See `docs/codeql-scan.md` H1–H3 + SPA. |
| Paramiko AutoAddPolicy | `py/paramiko-missing-host-key-validation` ×3 ✓ | — | CodeQL-only. |

The two scans are complementary: CodeQL's data-flow taint analysis catches the
path-traversal chain Semgrep missed; Semgrep's curated library rules catch the
second XXE parser and the IIIF SSRF that CodeQL's pack did not surface. Run
both.

## Actionable fix list (combined priority)

1. **XXE** — swap `xml.etree.ElementTree` → `defusedxml.ElementTree` at
   `session_manager.py:466` and `utils/__init__.py:134`; add `defusedxml` to
   `pyproject.toml`. *(fixes CodeQL M3 + Semgrep E1 + Semgrep E2 in one pass)*
2. **IIIF SSRF** — enforce `http(s)` scheme (and optional host allowlist) at
   `session_manager.py:666, 670`. *(Semgrep W1)*
3. **Path injection** — confine `import-xml`, C2AV plate, and SPA catch-all
   paths to their roots. *(CodeQL H1/H2/H3 + SPA — see `docs/codeql-scan.md`)*
4. **Paramiko host keys** — replace `AutoAddPolicy` with pinned keys.
   *(CodeQL M1/M2)*

Items 1 and 2 are bounded, single-file edits and good candidates for one
`@fixer` pass. Items 3 and 4 need the policy decisions noted in
`docs/codeql-scan.md`.

## Coverage notes

- Semgrep ran 755 total rules across two packs vs CodeQL's 45 Python queries.
  Semgrep parsed 99.9% of lines; 1538 files matched `.semgrepignore` (mostly
  `node_modules`, `frontend/dist`, `tmp/`) and 12 files >1 MB were skipped
  (the bundled PDFs).
- Semgrep found **no** command-injection, SQLi, hardcoded-secret, weak-crypto,
  insecure-cookie, or ReDoS findings — consistent with CodeQL's clean classes.
- The `p/owasp-top-ten` pack's 2 findings were a strict subset of `--config
  auto`'s 7 (both XXE). Running both adds no unique findings here, but
  `p/owasp-top-ten` is worth keeping in CI for regression gating.
- No syntax/parse errors were reported by either tool — all 56 Python files
  parsed cleanly. `--config auto` includes the `python.lang.best-practice` and
  `python.lang.maintainability` packs, which would surface syntax issues; none
  fired.
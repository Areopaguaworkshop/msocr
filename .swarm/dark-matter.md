## Dark Matter: Hidden Couplings

Found 7 file pairs that frequently co-change but have no import relationship:

| File A | File B | NPMI | Co-Changes | Lift |
|--------|--------|------|------------|------|
| instruction/Start-here.md | instruction/summary.md | 1.000 | 3 | 6.67 |
| instruction/Start-here.md | output/state_update.yaml | 1.000 | 3 | 6.67 |
| instruction/Start-here.md | pipeline/base_msocr_compatible.yaml | 1.000 | 3 | 6.67 |
| instruction/summary.md | output/state_update.yaml | 1.000 | 3 | 6.67 |
| instruction/summary.md | pipeline/base_msocr_compatible.yaml | 1.000 | 3 | 6.67 |
| output/state_update.yaml | pipeline/base_msocr_compatible.yaml | 1.000 | 3 | 6.67 |
| README.md | pyproject.toml | 0.609 | 4 | 2.67 |

These pairs likely share an architectural concern invisible to static analysis.
Consider adding explicit documentation or extracting the shared concern.
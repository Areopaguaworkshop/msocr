# Multi-Script vs. Per-Script HTR: Evidence and Recommendation

**Question:** Should we combine the three Sogdian script systems (Christian/Syriac, Sogdian national, Manichaean) into one Kraken HTR model, or train separate models per script? Broader goal: a system extensible to Syriac, Coptic, and other ancient scripts.

**Answer:** Separate models per script. Do not combine. HIGH confidence.

**Date:** 2026-06-24

---

## The Critical Distinction: Within-Script Variety vs. Cross-Script Mixing

The "variety > volume" principle (Reul 2022, Torres Aguilar 2023) is **within-script**: different hands, documents, or typefaces using the *same* script (same Unicode block, same glyph shapes). It does NOT extend to mixing different scripts with different glyph shapes.

### Within-script variety HELPS

- **Reul, Tomasek, Langhanki, Springmann (2022)** — Gothic + Bastarda (both Latin). Combined CER 6.22%. In-domain > combined > out-of-domain. [arXiv:2201.07661](https://arxiv.org/abs/2201.07661)
- **Torres Aguilar, Jolivet (2023)** — Textualis + Cursiva (both Latin). "Cross-training does not result in any loss of accuracy... leads to a slight improvement." [HAL:hal-03892163](https://hal.science/hal-03892163)
- **Torres Aguilar (2024)** — TRIDIS v2. "Blending families, hands and languages during training is a recommended strategy." [Zenodo:10.5281/zenodo.13862096](https://zenodo.org/records/13862096)
- **Hodel et al. (2021)** — 5.1M tokens, hundreds of hands, all German Kurrent. CER 6.53%. Variety prevents overfitting. [DOI:10.5334/johd.46](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.46)
- **Camps et al. (2022)** — "The variety of the training set increases robustness." (Latin script.) [DH2022](https://dh2022.dhii.asia/abstracts/files/CAMPS_Jean_Baptiste_Data_Diversity_in_handwritten_text_recog.html)

### Cross-script mixing HURTS or is UNPROVEN

- **Kraken docs (v2.0)** — "Multi-script models, e.g. combined polytonic Greek and Latin, will require significantly more transcriptions." [kraken.re/2.0.0/training.html](https://kraken.re/2.0.0/training.html)
- **Kiessling (2025), Kraken Issue #711** — Even mixing handwriting + printing *within one script* flagged as suboptimal. [github.com/mittagessen/kraken/issues/711](https://github.com/mittagessen/kraken/issues/711)
- **On the Generalization of HTR Models (2024)** — "Significant performance drop in OOD scenarios... linguistic divergence is the major factor." CRNNs do NOT generalize across script boundaries. [arXiv:2411.17332](https://arxiv.org/html/2411.17332v1)
- **Cross-Lingual Learning within Arabic Script (2025)** — Cross-lingual joint training helps *only because* all languages share Arabic glyphs. Improvements concentrate on shared characters. [arXiv:2605.02089](https://arxiv.org/html/2605.02089) — this is the ceiling for cross-script transfer, and it only works within one script.

### The synthesis

| Principle | Applies to | Does NOT apply to |
|---|---|---|
| "Variety > volume" | Different hands within same script | Different scripts, different glyphs |
| "Mixed models generalize better" | Multiple documents, same script | Syriac + Sogdian national + Manichaean |
| "Cross-training doesn't hurt" | Textualis + Cursiva (both Latin) | Different Unicode blocks |

---

## The Three Sogdian Scripts Share NO Glyph Shapes

| Script | Unicode Block | Glyph Set | Joining | Ancestry |
|---|---|---|---|---|
| Christian Sogdian | U+0700 (Syriac) | Syriac Estrangelo + 3-4 added letters | Syriac joining | Syriac/Aramaic |
| Sogdian national | U+10F30 (Sogdian) | Formal + cursive, 42 chars | Sogdian joining (Arabic-like) | Old Sogdian → Imperial Aramaic |
| Manichaean Sogdian | U+10AC0 (Manichaean) | 51 chars, Palmyrene-derived | Unique Left_Joining chars | Palmyrene Aramaic |

All three write the Sogdian language (RTL abjad, same phoneme inventory) but diverged into visually distinct scripts. A CRNN's CNN layers learn pixel→char mappings; forcing one output class (ALEPH) to map to three unrelated visual shapes confuses the encoder.

**What does transfer:** linguistic patterns (n-grams, plausible char sequences) in the LSTM layers — because all three write the same language. This enables **sequential transfer** (finetune one model from another), not joint training.

---

## Practical Trade-offs

- **Data sparsity / visual features:** NO cross-script borrowing. CNN learns per-script glyphs.
- **Data sparsity / linguistic features:** YES, but only via sequential finetuning. Not via joint training.
- **Kraken `--resize`:** `add` extends codec for new chars; technically can output multiple Unicode blocks, but the visual encoder can't learn which script it's seeing.
- **Inference cost:** Negligible difference between 1 vs. 3 models (5-20MB each, ms-scale inference). Don't let this drive the decision.

---

## Recommendation: Model Zoo with Sequential Transfer

### Strategy comparison

| Strategy | Evidence | Expected | Risk |
|---|---|---|---|
| (a) Per-script models | STRONG | Best per-script accuracy | Separate runs, no cross-script benefit |
| (b) Unified 3-script model | WEAK/NEGATIVE | Poor on all three; confused encoder | Wastes annotation budget |
| (c) Sequential transfer (one model → finetune next) | MODERATE | Good on primary; faster convergence on secondary | Less stable than within-script finetuning |

### Chosen: (a) with optional (c)

**Phase 1 — Christian Sogdian (now):** Finetune Sophro Mhiro Syriac base. All annotation here. Accept ~20-30% CER with hundreds of lines (assisted transcription, not autonomous).

**Phase 2 — Sogdian national script (future):** Finetune from Avestan base (already downloaded) or from the trained Christian Sogdian model via `--resize add`. LSTM retains Sogdian language knowledge; CNN adapts to new glyphs.

**Phase 3 — Manichaean (future):** Most visually distinct (unique Left_Joining). Finetune from Christian Sogdian or train from scratch.

**Beyond Sogdian (Syriac, Coptic, etc.):** Each script gets its own base + finetune path, registered in `language_registry.SCRIPT_BLOCKS` + `DEFAULT_BASE_MODELS`. Adding a language = add a registry entry, source a base model, run a finetune. Christian Sogdian's model doubles as a Syriac-script base for other Syriac-language work (Christian Palestinian Aramaic, Syriac proper) — a free win since it IS Syriac script.

### Why NOT a unified model

1. Different Unicode blocks = different glyphs. CNN must learn three unrelated visual vocabularies.
2. Data dilution: hundreds of lines / 3 scripts ≈ unusable for all three.
3. Zero published precedent for successful joint CRNN/CTC training on different scripts.
4. Kraken maintainer treats single-script as default; multi-script as data-hungry special case.
5. Generalization gap across scripts is a documented architectural limit, not a data problem.

---

## Key Citations

### Within-script variety (mixing helps WITHIN a script)
1. Reul et al. (2022). [arXiv:2201.07661](https://arxiv.org/abs/2201.07661)
2. Torres Aguilar & Jolivet (2023). [HAL:hal-03892163](https://hal.science/hal-03892163)
3. Torres Aguilar (2024). [Zenodo:10.5281/zenodo.13862096](https://zenodo.org/records/13862096)
4. Hodel et al. (2021). [DOI:10.5334/johd.46](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.46)
5. Camps et al. (2022). [DH2022](https://dh2022.dhii.asia/abstracts/files/CAMPS_Jean_Baptiste_Data_Diversity_in_handwritten_text_recog.html)

### Cross-script limitations
6. Kraken docs (v2.0). [kraken.re/2.0.0/training.html](https://kraken.re/2.0.0/training.html)
7. Kiessling (2025). [github.com/mittagessen/kraken/issues/711](https://github.com/mittagessen/kraken/issues/711)
8. On the Generalization of HTR Models (2024). [arXiv:2411.17332](https://arxiv.org/html/2411.17332v1)
9. Cross-Lingual Learning within Arabic Script (2025). [arXiv:2605.02089](https://arxiv.org/html/2605.02089)

### Transfer learning (sequential, not joint)
10. Vogler et al. (2022). Self-supervised pretraining → finetune on ~30 lines; cross-script. [ACL](https://aclanthology.org/2022.findings-naacl.15.pdf)
11. Granet et al. (2018). Transfer learning across domains/periods. [ACL](https://aclanthology.org/C18-1125.pdf)
12. Torres Aguilar (2024). "Transfer learning practices are increasingly prevalent." [HAL:hal-04716654](https://hal.science/hal-04716654/document)

### Sogdian script documentation
16. Unicode Sogdian (U+10F30). [Chart](https://www.unicode.org/charts/PDF/U10F30.pdf)
17. Unicode Manichaean (U+10AC0). [Chart](https://www.unicode.org/charts/PDF/U10AC0.pdf)
18. Everson & Durkin-Meisterernst (2011). Manichaean proposal. [Unicode L2/11-123R](https://www.unicode.org/L2/L2011/11123r-n4029r-manichaean.pdf)
19. Sims-Williams, N. "Sogdian Language and Its Scripts." [Smithsonian](https://sogdians.si.edu/sidebars/sogdian-language/)

---

## Bottom Line

| Question | Answer | Confidence |
|---|---|---|
| Combine all three Sogdian scripts in one model? | **No** | HIGH |
| Train Christian Sogdian model first? | **Yes** | HIGH |
| Use Christian model as base for finetuning other scripts later? | **Yes, if needed** | MODERATE |
| Will a unified model save annotation effort? | **No — it would waste it** | HIGH |
| Does "variety > volume" apply across scripts? | **No — within-script only** | HIGH |
| Extensibility model for Syriac/Coptic/etc.? | **Model zoo, per-script, sequential transfer** | HIGH |
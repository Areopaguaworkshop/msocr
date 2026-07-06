# c2av union codec (base + 5 new Sogdian classes)

Base model: `/home/ajiap/project/msocr/models/kraken/sophro_mhiro_syriac.mlmodel`
Base classes: **37** (row 0 = CTC blank, rows 1..36 = 36 chars)
Union classes: **42** (5 new rows appended)

## New classes (rows to LEAVE TRAINABLE in Phase 1)

| Row | Char | Codepoint | Name |
|---|---|---|---|
| 37 | ݁ | U+0741 | SYRIAC QUSHSHAYA |
| 38 | ݂ | U+0742 | SYRIAC RUKKAKHA |
| 39 | ݍ | U+074D | SYRIAC LETTER SOGDIAN ZHAIN |
| 40 | ݎ | U+074E | SYRIAC LETTER SOGDIAN KHAPH |
| 41 | ݏ | U+074F | SYRIAC LETTER SOGDIAN FE |

## Old classes (rows to FREEZE in Phase 1): 37 rows

| Row | Char |
|---|---|
| 0 | <BLANK> |
| 1 | ' ' |
| 2 | . |
| 3 | ̈ |
| 4 | ̱ |
| 5 | ܐ |
| 6 | ܒ |
| 7 | ܓ |
| 8 | ܕ |
| 9 | ܗ |
| 10 | ܘ |
| 11 | ܙ |
| 12 | ܚ |
| 13 | ܛ |
| 14 | ܝ |
| 15 | ܟ |
| 16 | ܠ |
| 17 | ܡ |
| 18 | ܢ |
| 19 | ܣ |
| 20 | ܥ |
| 21 | ܦ |
| 22 | ܨ |
| 23 | ܩ |
| 24 | ܪ |
| 25 | ܫ |
| 26 | ܬ |
| 27 | ( |
| 28 | ) |
| 29 | : |
| 30 | ? |
| 31 | [ |
| 32 | ] |
| 33 | ̄ |
| 34 | ̇ |
| 35 | ܀ |
| 36 | ܤ |

## Source / cross-check

5 new classes match `NEW_CLASSES` in `scripts/confusion_analyze.py:27`:
`{ݎ, ݏ, ݍ, SYRIAC QUSHSHAYA, SYRIAC RUKKAKHA}`.

Row 0 is the CTC blank (always present, never trained).

# Model Storage

Place OCR/HTR model files in this folder.

Current convention:
- Greek printed Kraken primary model:
  `models/kraken/greek-english_porson_sophoclesplaysa05campgoog/greek-english_porson_sophoclesplaysa05campgoog.mlmodel`
- Greek printed Kraken fallback models:
  - `models/kraken/greek-german_serifs_sophokle1v3soph/greek-german_serifs_sophokle1v3soph.mlmodel`
  - `models/kraken/greek-german_serifs_bsb10234118/greek-german_serifs_bsb10234118.mlmodel`
- Latin printed Kraken primary model (CATMuS-Print Large):
  - `models/kraken/latin_printed_catmus_large.mlmodel`
- Latin handwritten Kraken model (McCATMuS):
  - `models/kraken/latin_handwritten_mccatmus.mlmodel`
- Greek handwritten default model (temporary Kraken fallback):
  - `models/kraken/greek-german_serifs_sophokle1v3soph/greek-german_serifs_sophokle1v3soph.mlmodel`
- Syriac printed custom Tesseract model slots (optional CER-triggered fallback):
  - `models/tesseract/syr_serto.traineddata`
  - `models/tesseract/syr_east.traineddata`
- Other language/model files: `models/<language>_<mode>.mlmodel`

Notes:
- Latin printed OCR defaults to Kraken CATMuS-Print Large and falls back to Tesseract.
- Syriac printed OCR defaults to Tesseract (`syr`), with optional CER-gated switch for Serto/East custom traineddata files.
- Syriac handwritten OCR is currently routed to external Transkribus workflow.
- Coptic printed OCR uses Tesseract only (OCRopy fallback is deactivated).
- Handwritten and custom printed Kraken routes can use explicit `--model` paths.

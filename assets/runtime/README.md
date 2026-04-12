The files in this directory are canonical runtime smoke assets for deployment validation.

`syriac_estrangela_smoke.png` is a small Syriac printed sample used by the live `/ocr`
runtime probe. It is intended to prove that the promoted printed model can be resolved
and serve a non-empty OCR response through the FastAPI endpoint; it is not a benchmark
fixture and should not be used for accuracy claims.

`syriac_estrangela_smoke.txt` records the source phrase used to render the sample image.

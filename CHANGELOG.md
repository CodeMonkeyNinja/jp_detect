# Changelog

## [1.0.0] — 2026-07-09

- **Stable API milestone** — no breaking changes planned. Core detection pipeline,
  parameter selection, and confidence scoring are mature and production-tested
  within the Lenzu ecosystem.

## [0.2.4] — 2026-05-xx

- Dependency security updates (openssl, imageproc, rand).
- CI: gate publish on passing tests; auto-publish on tag push.

## [0.2.3] — 2026-05-xx

- **Orientation-aware merging** — prevents vertical/horizontal cross-merging
  of text regions regardless of pad size.
- **Graduated padding scale** — padding increases for larger images
  (inter-line gaps grow with resolution).

## [0.2.2] — 2026-04-xx

- Exclude `examples/output/` from crate package.

## [0.2.1] — 2026-04-xx

- **Confidence scores** — per-box mean probability (0.0–1.0) from DBNet
  probability map.
- **Contour polygons** — `Vec<Vec<[u32; 2]>>` per box for oriented bounding
  boxes and text-angle estimation.
- **`detect_with_map()`** — returns raw 640×640 probability map alongside
  detection boxes.
- **Scale table** — `detection_params_for_size()` selects threshold,
  dilation, and padding from a built-in table based on image dimensions.

## [0.2.0] — 2026-04-xx

- Bundle StabRise `text_detection_dbnet_ml_v0.2` ONNX model (~4.7 MB).
- `detect_and_draw` example with annotated output.
- Exclude `tests/fixtures/` from crate package.

## [0.1.0] — 2026-04-xx

- Initial release: DBNet text detection with ONNX Runtime via `ort`.

# jp_detect

Japanese text detection bounding-box library.

Real-time scene text detection using the [DBNet](https://arxiv.org/abs/1911.08947)
architecture and an embedded ONNX model.

The bundled model is the StabRise multi-language variant
[`text_detection_dbnet_ml_v0.2`](https://huggingface.co/StabRise/text_detection_dbnet_ml_v0.2)
(~4.7 MB, input 640 × 640).  It detects text in any orientation, including
vertical Japanese columns.

## Usage

```toml
[dependencies]
jp_detect = { version = "1", features = ["onnx"] }
```

```rust
use image::GenericImageView;
use jp_detect::{build_text_detector, detection_params_for_size};

let image = image::open("screenshot.png")?;
let (w, h) = image.dimensions();
let p = detection_params_for_size(w, h);
let detector = build_text_detector(None, p.threshold, p.dilation, p.pad_x, p.pad_y)?
    .unwrap();
for b in detector.detect(&image) {
    println!("[{:.0}%] ({},{})–({},{})", b.confidence * 100.0,
             b.x1, b.y1, b.x2, b.y2);
}
```

`detection_params_for_size` selects threshold, dilation, and padding from a
built-in scale table based on the image's longest edge (see **Scale table**
below).  Pass a file path as the first argument to `build_text_detector` to use
a custom model instead of the bundled one.

## Examples

### `detect_and_draw`

Converts each input image to greyscale, detects text regions, draws red
bounding boxes over them, and writes the annotated PNG and a JSON file with
box coordinates to `/dev/shm/jp_detect/`.

```sh
# Run on the built-in test fixtures
cargo run --example detect_and_draw --features onnx

# Run on your own images
cargo run --example detect_and_draw --features onnx -- /path/to/a.png /path/to/b.png
```

Output per image (e.g. for `screenshot.png`):

```
image : /dev/shm/jp_detect/screenshot_annotated.png
json  : /dev/shm/jp_detect/screenshot_boxes.json (3 boxes)
```

JSON format:

```json
[
  {"index": 0, "x1": 3, "y1": 0, "x2": 178, "y2": 349, "width": 175, "height": 349, "confidence": 0.9745, "contours": 1, "contour_points": 1235},
  {"index": 1, "x1": 142, "y1": 23, "x2": 630, "y2": 233, "width": 488, "height": 210, "confidence": 0.9923, "contours": 2, "contour_points": 1844}
]
```

See [unified OCR benchmark](https://github.com/HidekiAI/lenzu/blob/trunk/docs/scores.md) for detection + OCR accuracy results across all engines.

## Pipeline

```
DynamicImage
  ↓ resize_exact(640×640, Triangle filter)
  ↓ normalize per channel: (px − mean) / std   ← ImageNet constants
  ↓ layout: CHW float32 tensor [1, 3, 640, 640]
  ↓
  ▼ ONNX inference (DBNet)
  ↓ probability map [1, 1, 640, 640]
  ↓ threshold → binary mask
  ↓ morphological dilation (Norm::L1)
  ↓ find_contours → AABB + polygon per contour
  ↓ per-contour confidence = mean(prob | prob ≥ threshold)
  ↓ scale contour + bbox back to original coordinates + pad
  ▼ union-merge overlapping boxes → Vec<TextBoundingBox>

  ⤷ detect_with_map() also returns the raw 640×640 probability map
```

## Feature flags

| Flag   | Description |
|--------|-------------|
| `onnx` | Enables `DbNetDetector` and embeds the ONNX model (~4.7 MB). Without this flag the public types are still compiled but `build_text_detector` always returns `Ok(None)`. |

## Scale table

`detection_params_for_size(w, h)` returns the appropriate parameters from the
built-in scale table based on the image's longest edge.  DBNet always runs at
640 × 640 internally, so dilation of *N* pixels at that resolution represents
*N* × (original / 640) pixels in the original — much more morphological blur
for large inputs.  Padding increases for larger images because inter-line gaps
(in original coordinates) grow with resolution.  Orientation-aware merging
prevents vertical/horizontal cross-merging regardless of pad size.

| Longest edge | Dilation | Threshold | Pad |
|--------------|----------|-----------|-----|
| ≤ 800        | 16       | 0.20      | 32  |
| ≤ 1 280      | 10       | 0.25      | 32  |
| ≤ 1 920      |  6       | 0.35      | 32  |
| ≤ 2 560      |  3       | 0.45      | 40  |
| > 2 560      |  0       | 0.50      | 48  |

You can also pass parameters directly to `build_text_detector` if you need
custom values.

## Parameter guidance

| Parameter  | Default | Notes |
|------------|---------|-------|
| `threshold`| `0.2`   | Lower than the paper's 0.3 default; the StabRise model produces sparser probability maps on Japanese text. |
| `dilation` | `16`    | At 640 × 640 scale ≈ 4 % of image width; merges per-character blobs into word/line regions. |
| `pad_x/y`  | `32`    | Applied at original-image scale; provides ascender/descender headroom. |
| `confidence` | — | Per-box mean probability (0.0–1.0) of thresholded pixels from the DBNet probability map. Higher = stronger model belief that the region contains text. |
| `contours` | — | Polygon vertices (`Vec<Vec<[u32; 2]>>`) in original-image coordinates. Useful for oriented bounding boxes, text-angle estimation, or tighter masking. |

### `detect_with_map`

`DbNetDetector::detect_with_map` returns a `DetectionOutput` containing the
bounding boxes **and** the raw 640 × 640 probability map.  Use this when you
need custom thresholding, heatmap visualisation, or any post-processing beyond
what the built-in pipeline provides.

## Available but unexposed data

The DBNet pipeline produces additional data that is not currently returned but
is documented in-code (`src/lib.rs`, `postprocess`) for future consideration.
Each item is marked with a `NOTE (unsupported data — ...)` comment.

| Data | Where available | Potential use |
|------|-----------------|---------------|
| Pre-dilation contours | Before `dilate()` call | Individual character/cluster blobs; rough character count; line segmentation |
| Max probability per region | `prob_map` max within contour bbox | Ranking detections by strongest pixel instead of mean |
| Fill ratio | `prob_count / bbox_area` at 640×640 scale | Distinguishing dense paragraphs from sparse labels |
| Thresholded pixel count | `prob_count` in confidence loop | Text-ink area proxy; rough measure of text quantity |
| Contour border type | `contour.border_type` (Outer/Hole) | Filtering false positives from interior boundaries |
| Contour parent hierarchy | `contour.parent` (index into contour vec) | Nested region detection (text inside bordered panels) |
| Discarded contour count | Contours with < 4 pts or < 5×5 bbox | Diagnostic signal for noisy input or aggressive threshold |

## Credits & Citations

### Text detection model

**Model variant:** [`text_detection_dbnet_ml_v0.2`](https://huggingface.co/StabRise/text_detection_dbnet_ml_v0.2)  
**Optimised by:** [StabRise](https://huggingface.co/StabRise)  
**Source:** Hugging Face

### Original DBNet research

```bibtex
@inproceedings{liao2020real,
  title     = {Real-time Scene Text Detection with Differentiable Binarization},
  author    = {Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2020}
}
```

## Dog-fooded in production

This crate is a core dependency of [Lenzu](https://github.com/CodeMonkeyNinja/lenzu) — a transparent OCR lens overlay for Linux desktop. It is pulled from crates.io and used at runtime for text detection before OCR.

## License

MIT — see [LICENSE](LICENSE).

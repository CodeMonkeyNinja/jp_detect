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
jp_detect = { git = "https://github.com/codemonkeyninja/jp_detect", features = ["onnx"] }
```

```rust
use jp_detect::build_text_detector;

// None → use the bundled model
let detector = build_text_detector(None, 0.2, 16, 32, 32)?.unwrap();
let image = image::open("screenshot.png")?;
for b in detector.detect(&image) {
    println!("({},{})–({},{})", b.x1, b.y1, b.x2, b.y2);
}
```

Pass a file path as the first argument to use a custom model instead of the
bundled one.

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
  {"index": 0, "x1": 122, "y1": 42, "x2": 670, "y2": 1494, "width": 548, "height": 1452},
  {"index": 1, "x1": 734, "y1": 213, "x2": 2668, "y2": 440, "width": 1934, "height": 227}
]
```

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
  ↓ find_contours → AABB per contour
  ↓ scale back to original coordinates + pad
  ▼ union-merge overlapping boxes → Vec<TextBoundingBox>
```

## Feature flags

| Flag   | Description |
|--------|-------------|
| `onnx` | Enables `DbNetDetector` and embeds the ONNX model (~4.7 MB). Without this flag the public types are still compiled but `build_text_detector` always returns `Ok(None)`. |

## Parameter guidance

| Parameter  | Default | Notes |
|------------|---------|-------|
| `threshold`| `0.2`   | Lower than the paper's 0.3 default; the StabRise model produces sparser probability maps on Japanese text. |
| `dilation` | `16`    | At 640 × 640 scale ≈ 4 % of image width; merges per-character blobs into word/line regions. |
| `pad_x/y`  | `32`    | Applied at original-image scale; provides ascender/descender headroom. |

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

## License

MIT — see [LICENSE](LICENSE).

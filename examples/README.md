# examples

## detect_and_draw

Converts each input image to greyscale, runs the embedded DBNet model to locate
text regions, draws a 2-pixel red bounding box around each region, and writes
the results to `/dev/shm/jp_detect/`.

Detection parameters (threshold, dilation, padding) are chosen automatically
per image via `detection_params_for_size(w, h)`, which selects from the built-in
scale table based on the image's longest edge.  See the main
[README](../README.md#scale-table) for the full table.

### Build

```sh
cargo build --example detect_and_draw --features onnx
```

### Run

**Default** — uses the two built-in test fixtures:

```sh
cargo run --example detect_and_draw --features onnx
```

**Custom images** — pass one or more paths as arguments:

```sh
cargo run --example detect_and_draw --features onnx -- /path/to/img1.png /path/to/img2.png
```

### Output

For each input image two files are written to `/dev/shm/jp_detect/`:

| File | Description |
|------|-------------|
| `<stem>_annotated.png` | Greyscale image with red bounding boxes drawn over detected text regions |
| `<stem>_boxes.json` | JSON array of detected bounding boxes |

Example terminal output:

```
image : /dev/shm/jp_detect/Unit-test-sample-texts_annotated.png
json  : /dev/shm/jp_detect/Unit-test-sample-texts_boxes.json (2 boxes)

image : /dev/shm/jp_detect/OCR-Demo-JP2EN_annotated.png
json  : /dev/shm/jp_detect/OCR-Demo-JP2EN_boxes.json (3 boxes)
```

### JSON format

```json
[
  {"index": 0, "x1": 3, "y1": 0, "x2": 178, "y2": 349, "width": 175, "height": 349, "confidence": 0.9745, "contours": 1, "contour_points": 1235},
  {"index": 1, "x1": 142, "y1": 23, "x2": 630, "y2": 233, "width": 488, "height": 210, "confidence": 0.9923, "contours": 2, "contour_points": 1844}
]
```

Each entry contains:

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Zero-based detection order (sorted top-to-bottom, left-to-right) |
| `x1`, `y1` | int | Top-left corner in original image coordinates (pixels) |
| `x2`, `y2` | int | Bottom-right corner in original image coordinates (pixels) |
| `width` | int | `x2 − x1` |
| `height` | int | `y2 − y1` |
| `confidence` | float | Mean probability (0.0–1.0) of thresholded pixels from DBNet probability map |
| `contours` | int | Number of contour polygons (>1 when boxes were merged) |
| `contour_points` | int | Total vertex count across all contour polygons |

### Sample output

**Unit-test-sample-texts.png** — 2 text regions detected (tategaki separate, yokogaki+tegaki merged):

![Unit-test-sample-texts annotated](output/Unit-test-sample-texts_annotated.png)

**OCR-Demo-JP2EN.png** — 3 text regions detected (multi-line dialogue merged):

![OCR-Demo-JP2EN annotated](output/OCR-Demo-JP2EN_annotated.png)

### Accessing the raw probability map

The example uses the `TextDetector::detect` trait method, which returns only
bounding boxes.  For access to the full 640 × 640 probability heatmap, use
`DbNetDetector::detect_with_map` instead:

```rust
use jp_detect::DbNetDetector;

let detector = DbNetDetector::from_bytes(model_bytes, 0.2, 16, 32, 32)?;
let output = detector.detect_with_map(&image);
// output.boxes            — Vec<TextBoundingBox>
// output.probability_map  — Vec<f32>, 640×640, row-major
// output.map_width        — 640
// output.map_height       — 640
```

### Observe

Open the annotated PNG with any image viewer:

```sh
xdg-open /dev/shm/jp_detect/Unit-test-sample-texts_annotated.png
xdg-open /dev/shm/jp_detect/OCR-Demo-JP2EN_annotated.png
```

Pretty-print the JSON with [`jq`](https://jqlang.org) or pipe it through
[`json_pp`](https://crates.io/crates/json_pp):

```sh
# using the json_pp crate (cargo install json_pp)
json_pp < /dev/shm/jp_detect/Unit-test-sample-texts_boxes.json
```

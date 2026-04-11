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
json  : /dev/shm/jp_detect/Unit-test-sample-texts_boxes.json (4 boxes)

image : /dev/shm/jp_detect/OCR-Demo-JP2EN_annotated.png
json  : /dev/shm/jp_detect/OCR-Demo-JP2EN_boxes.json (5 boxes)
```

### JSON format

```json
[
  {"index": 0, "x1": 122, "y1": 42,  "x2": 670,  "y2": 1494, "width": 548,  "height": 1452, "confidence": 0.9920, "contours": 1, "contour_points": 142},
  {"index": 1, "x1": 734, "y1": 213, "x2": 2668, "y2": 440,  "width": 1934, "height": 227,  "confidence": 0.9959, "contours": 1, "contour_points": 56},
  {"index": 2, "x1": 742, "y1": 599, "x2": 2650, "y2": 915,  "width": 1908, "height": 316,  "confidence": 0.9970, "contours": 1, "contour_points": 78}
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

**Unit-test-sample-texts.png** — 4 text regions detected:

![Unit-test-sample-texts annotated](output/Unit-test-sample-texts_annotated.png)

**OCR-Demo-JP2EN.png** — 5 text regions detected:

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

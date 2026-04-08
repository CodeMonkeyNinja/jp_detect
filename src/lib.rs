//! # jp_detect
//!
//! Real-time scene text detection backed by the
//! [DBNet](https://arxiv.org/abs/1911.08947) architecture and an ONNX runtime session.
//!
//! The bundled model is the StabRise multi-language variant
//! ([`text_detection_dbnet_ml_v0.2`](https://huggingface.co/StabRise/text_detection_dbnet_ml_v0.2)).
//! It runs at 640 × 640 and detects text in any orientation, including vertical
//! Japanese columns.
//!
//! ## Feature flags
//!
//! | Flag   | What it enables |
//! |--------|-----------------|
//! | `onnx` | [`DbNetDetector`], [`build_text_detector`], and the bundled ONNX model (~4.7 MB added to your binary). Without this flag the public types are still available but `build_text_detector` always returns `Ok(None)`. |
//!
//! ## Quick start
//!
//! ```no_run
//! # #[cfg(feature = "onnx")]
//! # {
//! use jp_detect::build_text_detector;
//!
//! // `None` → use the bundled model (requires the `onnx` feature).
//! let detector = build_text_detector(None, 0.2, 16, 32, 32).unwrap().unwrap();
//! let image = image::open("screenshot.png").unwrap();
//! let boxes = detector.detect(&image);
//! for b in &boxes {
//!     println!("text region: ({},{})–({},{})", b.x1, b.y1, b.x2, b.y2);
//! }
//! # }
//! ```
//!
//! ## References
//!
//! - Liao, M., Wan, Z., Yao, C., Chen, K., & Bai, X. (2020).
//!   *Real-Time Scene Text Detection with Differentiable Binarization.*
//!   AAAI Conference on Artificial Intelligence.
//!   <https://arxiv.org/abs/1911.08947>
//!
//! - StabRise. *text_detection_dbnet_ml_v0.2* (ONNX, multi-language).
//!   Hugging Face. <https://huggingface.co/StabRise/text_detection_dbnet_ml_v0.2>

use anyhow::Result;
use image::DynamicImage;
#[cfg(feature = "onnx")]
use image::GenericImageView;

// ── Public types (always compiled, no feature gate) ───────────────────────────

/// Axis-aligned bounding box in the coordinate space of the source image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextBoundingBox {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}

impl TextBoundingBox {
    pub fn width(&self) -> u32 {
        self.x2.saturating_sub(self.x1)
    }

    pub fn height(&self) -> u32 {
        self.y2.saturating_sub(self.y1)
    }

    /// True when any part of `self` overlaps `other`.
    pub fn intersects(&self, other: &Self) -> bool {
        self.x1 < other.x2 && self.x2 > other.x1 && self.y1 < other.y2 && self.y2 > other.y1
    }
}

/// Swappable text-detection backend.
///
/// Implement this trait to plug in an alternative model or a mock detector for
/// testing.
pub trait TextDetector: Send + Sync {
    fn detect(&self, image: &DynamicImage) -> Vec<TextBoundingBox>;
}

/// Compute the axis-aligned union of all bounding boxes.
///
/// Returns `None` if `boxes` is empty.
pub fn compute_union_bbox(boxes: &[TextBoundingBox]) -> Option<TextBoundingBox> {
    if boxes.is_empty() {
        return None;
    }
    let x1 = boxes.iter().map(|b| b.x1).min().unwrap();
    let y1 = boxes.iter().map(|b| b.y1).min().unwrap();
    let x2 = boxes.iter().map(|b| b.x2).max().unwrap();
    let y2 = boxes.iter().map(|b| b.y2).max().unwrap();
    Some(TextBoundingBox { x1, y1, x2, y2 })
}

/// Return the index of the box whose nearest edge is closest to `(px, py)`.
///
/// If the cursor is inside a box, that box has distance 0 and is always
/// preferred.  Uses squared distance — no `sqrt` required.
///
/// # Panics
///
/// Panics if `boxes` is empty.
pub fn closest_box_to_point(boxes: &[TextBoundingBox], px: u32, py: u32) -> usize {
    assert!(!boxes.is_empty(), "closest_box_to_point called with empty slice");
    boxes
        .iter()
        .enumerate()
        .min_by_key(|(_, b)| {
            let nx = (px as i64).clamp(b.x1 as i64, b.x2 as i64);
            let ny = (py as i64).clamp(b.y1 as i64, b.y2 as i64);
            let dx = nx - px as i64;
            let dy = ny - py as i64;
            dx * dx + dy * dy
        })
        .unwrap()
        .0
}

// ── Bundled model (onnx feature only) ─────────────────────────────────────────

/// The StabRise `text_detection_dbnet_ml_v0.2` ONNX model, baked in at compile
/// time.  Only present when the `onnx` feature is enabled.
///
/// End-users who prefer a smaller binary or a custom model can call
/// [`build_text_detector`] with an explicit path instead.
#[cfg(feature = "onnx")]
static EMBEDDED_MODEL: &[u8] = include_bytes!(
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/stabrise-text_detection_dbnet_ml_v02_model.onnx"
    )
);

// ── DBNet constants ────────────────────────────────────────────────────────────

#[cfg(feature = "onnx")]
const INPUT_SIZE: u32 = 640;

/// ImageNet per-channel means (R, G, B).
#[cfg(feature = "onnx")]
const MEAN: [f32; 3] = [123.675, 116.28, 103.53];

/// ImageNet per-channel standard deviations (R, G, B).
#[cfg(feature = "onnx")]
const STD: [f32; 3] = [58.395, 57.12, 57.375];

/// Gap tolerance for the merge pass (pixels).
///
/// Two boxes that are ≤ 4 px apart are considered touching.  This absorbs
/// 1–2 px rounding noise from the probability map without accidentally merging
/// genuinely separate regions (which are always ≫ 4 px apart).
#[cfg(feature = "onnx")]
const MERGE_GAP: u32 = 4;

// ── DbNetDetector ─────────────────────────────────────────────────────────────

/// DBNet text detector backed by an ONNX runtime session.
///
/// # Model
///
/// The default model is the StabRise multi-language variant of DBNet
/// (`text_detection_dbnet_ml_v0.2`).  Input: `[1, 3, 640, 640]` float32
/// (ImageNet-normalised CHW).  Output: `[1, 1, 640, 640]` probability map.
///
/// ## Parameter guidance
///
/// | Parameter  | Default | Notes |
/// |------------|---------|-------|
/// | `threshold`| `0.2`   | StabRise's model is sparser than the DBNet paper's 0.3 default on Japanese text; 0.2 recovers more mass. |
/// | `dilation` | `16`    | At 640 × 640 scale ≈ 4 % of image width; merges per-character blobs into word/line regions. |
/// | `pad_x`    | `32`    | Extra pixels added to each detected box (in original-image coordinates). |
/// | `pad_y`    | `32`    | Extra pixels added to each detected box (in original-image coordinates). |
///
/// # References
///
/// - Liao et al. (2020). *Real-Time Scene Text Detection with Differentiable
///   Binarization.* AAAI. <https://arxiv.org/abs/1911.08947>
/// - StabRise. *text_detection_dbnet_ml_v0.2.*
///   <https://huggingface.co/StabRise/text_detection_dbnet_ml_v0.2>
#[cfg(feature = "onnx")]
pub struct DbNetDetector {
    /// ONNX session wrapped in a `Mutex` because `ort` requires `&mut Session`
    /// for `run()`.
    session: std::sync::Mutex<ort::session::Session>,
    input_name: String,
    threshold: f32,
    dilation: u8,
    pad_x: u32,
    pad_y: u32,
}

#[cfg(feature = "onnx")]
impl DbNetDetector {
    /// Load from a file path.
    pub fn new(
        model_path: &str,
        threshold: f32,
        dilation: u8,
        pad_x: u32,
        pad_y: u32,
    ) -> Result<Self> {
        let session = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("ort session builder failed: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("failed to load ONNX model from '{model_path}': {e}"))?;
        Self::from_session(session, threshold, dilation, pad_x, pad_y)
    }

    /// Load from in-memory bytes.
    ///
    /// Used internally by [`build_text_detector`] when `model_path` is `None`
    /// (i.e. to load the [`EMBEDDED_MODEL`]).  Can also be used directly if you
    /// store model bytes yourself.
    pub fn from_bytes(
        model_bytes: &[u8],
        threshold: f32,
        dilation: u8,
        pad_x: u32,
        pad_y: u32,
    ) -> Result<Self> {
        let session = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("ort session builder failed: {e}"))?
            .commit_from_memory(model_bytes)
            .map_err(|e| anyhow::anyhow!("failed to load ONNX model from memory: {e}"))?;
        Self::from_session(session, threshold, dilation, pad_x, pad_y)
    }

    fn from_session(
        session: ort::session::Session,
        threshold: f32,
        dilation: u8,
        pad_x: u32,
        pad_y: u32,
    ) -> Result<Self> {
        let input_name = session.inputs()[0].name().to_string();
        Ok(Self {
            session: std::sync::Mutex::new(session),
            input_name,
            threshold,
            dilation,
            pad_x,
            pad_y,
        })
    }
}

#[cfg(feature = "onnx")]
impl TextDetector for DbNetDetector {
    fn detect(&self, image: &DynamicImage) -> Vec<TextBoundingBox> {
        let (orig_w, orig_h) = image.dimensions();
        let flat = preprocess(image);

        let shape = [1usize, 3, INPUT_SIZE as usize, INPUT_SIZE as usize];
        let input_tensor = match ort::value::Tensor::<f32>::from_array((shape, flat)) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[jp_detect] failed to create input tensor: {e}");
                return vec![];
            }
        };

        let mut session = self.session.lock().unwrap();
        let outputs =
            match session.run(ort::inputs![self.input_name.as_str() => input_tensor]) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("[jp_detect] inference failed: {e}");
                    return vec![];
                }
            };

        let prob_vec = match outputs[0].try_extract_tensor::<f32>() {
            Ok((_, slice)) => slice.to_vec(),
            Err(e) => {
                eprintln!("[jp_detect] failed to extract output tensor: {e}");
                return vec![];
            }
        };

        postprocess(
            &prob_vec,
            orig_w,
            orig_h,
            self.threshold,
            self.dilation,
            self.pad_x,
            self.pad_y,
        )
    }
}

// ── build_text_detector ────────────────────────────────────────────────────────

/// Construct a text detector.
///
/// | `model_path`  | `onnx` feature | Result |
/// |---------------|----------------|--------|
/// | `None`        | enabled        | `Ok(Some(_))` — uses the bundled ONNX model |
/// | `Some(path)`  | enabled        | `Ok(Some(_))` — loads model from `path` |
/// | `None`        | disabled       | `Ok(None)` — detection silently disabled |
/// | `Some(_)`     | disabled       | `Ok(None)` + warning logged to stderr |
///
/// # Parameters
///
/// See [`DbNetDetector`] for parameter guidance.
pub fn build_text_detector(
    model_path: Option<&str>,
    threshold: f32,
    dilation: u8,
    pad_x: u32,
    pad_y: u32,
) -> Result<Option<std::sync::Arc<dyn TextDetector + Send + Sync>>> {
    #[cfg(feature = "onnx")]
    {
        let detector = match model_path {
            Some(path) => DbNetDetector::new(path, threshold, dilation, pad_x, pad_y)?,
            None => DbNetDetector::from_bytes(EMBEDDED_MODEL, threshold, dilation, pad_x, pad_y)?,
        };
        return Ok(Some(
            std::sync::Arc::new(detector) as std::sync::Arc<dyn TextDetector + Send + Sync>,
        ));
    }

    #[cfg(not(feature = "onnx"))]
    {
        if model_path.is_some() {
            eprintln!(
                "[jp_detect] text_detection_model is configured but this binary was built \
                 without the `onnx` feature — text detection disabled"
            );
        }
        let _ = (threshold, dilation, pad_x, pad_y);
        Ok(None)
    }
}

// ── Private helpers ────────────────────────────────────────────────────────────

#[cfg(feature = "onnx")]
fn preprocess(img: &DynamicImage) -> Vec<f32> {
    let resized = img.resize_exact(INPUT_SIZE, INPUT_SIZE, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let n = INPUT_SIZE as usize;
    let mut data = vec![0f32; 3 * n * n];
    for y in 0..n {
        for x in 0..n {
            let px = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * n * n + y * n + x] = (px[c] as f32 - MEAN[c]) / STD[c];
            }
        }
    }
    data
}

#[cfg(feature = "onnx")]
fn postprocess(
    prob_map: &[f32],
    orig_w: u32,
    orig_h: u32,
    threshold: f32,
    dilation: u8,
    pad_x: u32,
    pad_y: u32,
) -> Vec<TextBoundingBox> {
    use image::{GrayImage, Luma};
    use imageproc::contours::find_contours;
    use imageproc::distance_transform::Norm;
    use imageproc::morphology::dilate;

    // Threshold → binary mask
    let mut gray = GrayImage::new(INPUT_SIZE, INPUT_SIZE);
    for y in 0..INPUT_SIZE as usize {
        for x in 0..INPUT_SIZE as usize {
            let v = if prob_map[y * INPUT_SIZE as usize + x] >= threshold {
                255
            } else {
                0
            };
            gray.put_pixel(x as u32, y as u32, Luma([v]));
        }
    }

    // Dilation: merge nearby blobs, compensate for DBNet's shrunk training targets
    let mask = if dilation > 0 {
        dilate(&gray, Norm::L1, dilation)
    } else {
        gray
    };

    let contours = find_contours::<u32>(&mask);
    let scale_x = orig_w as f32 / INPUT_SIZE as f32;
    let scale_y = orig_h as f32 / INPUT_SIZE as f32;

    let mut boxes = Vec::new();
    for contour in &contours {
        if contour.points.len() < 4 {
            continue;
        }
        let min_x = contour.points.iter().map(|pt| pt.x).min().unwrap();
        let max_x = contour.points.iter().map(|pt| pt.x).max().unwrap();
        let min_y = contour.points.iter().map(|pt| pt.y).min().unwrap();
        let max_y = contour.points.iter().map(|pt| pt.y).max().unwrap();

        if max_x.saturating_sub(min_x) < 5 || max_y.saturating_sub(min_y) < 5 {
            continue;
        }

        // Scale back to original image coordinates
        let x1 = (min_x as f32 * scale_x).round() as u32;
        let y1 = (min_y as f32 * scale_y).round() as u32;
        let x2 = (max_x as f32 * scale_x).round() as u32;
        let y2 = (max_y as f32 * scale_y).round() as u32;

        // Expand by pad_x / pad_y, clamped to image bounds
        let x1 = x1.saturating_sub(pad_x);
        let y1 = y1.saturating_sub(pad_y);
        let x2 = (x2 + pad_x).min(orig_w);
        let y2 = (y2 + pad_y).min(orig_h);

        boxes.push(TextBoundingBox { x1, y1, x2, y2 });
    }

    merge_overlapping(boxes)
}

#[cfg(feature = "onnx")]
fn overlaps(a: &TextBoundingBox, b: &TextBoundingBox) -> bool {
    a.x1 < b.x2 + MERGE_GAP
        && a.x2 + MERGE_GAP > b.x1
        && a.y1 < b.y2 + MERGE_GAP
        && a.y2 + MERGE_GAP > b.y1
}

#[cfg(feature = "onnx")]
fn union_of(a: &TextBoundingBox, b: &TextBoundingBox) -> TextBoundingBox {
    TextBoundingBox {
        x1: a.x1.min(b.x1),
        y1: a.y1.min(b.y1),
        x2: a.x2.max(b.x2),
        y2: a.y2.max(b.y2),
    }
}

/// Iteratively merge overlapping boxes until no overlaps remain, then sort
/// top-to-bottom, left-to-right.
#[cfg(feature = "onnx")]
fn merge_overlapping(mut boxes: Vec<TextBoundingBox>) -> Vec<TextBoundingBox> {
    loop {
        let mut merged = Vec::with_capacity(boxes.len());
        let mut consumed = vec![false; boxes.len()];
        let mut any = false;

        for i in 0..boxes.len() {
            if consumed[i] {
                continue;
            }
            let mut current = boxes[i].clone();
            for j in (i + 1)..boxes.len() {
                if consumed[j] {
                    continue;
                }
                if overlaps(&current, &boxes[j]) {
                    current = union_of(&current, &boxes[j]);
                    consumed[j] = true;
                    any = true;
                }
            }
            merged.push(current);
        }

        boxes = merged;
        if !any {
            break;
        }
    }
    boxes.sort_by(|a, b| a.y1.cmp(&b.y1).then(a.x1.cmp(&b.x1)));
    boxes
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(x1: u32, y1: u32, x2: u32, y2: u32) -> TextBoundingBox {
        TextBoundingBox { x1, y1, x2, y2 }
    }

    // ── Pure-logic tests (no `onnx` feature needed) ────────────────────────

    #[test]
    fn test_compute_union_empty() {
        assert!(compute_union_bbox(&[]).is_none());
    }

    #[test]
    fn test_compute_union_single() {
        let b = compute_union_bbox(&[bbox(10, 20, 30, 40)]).unwrap();
        assert_eq!((b.x1, b.y1, b.x2, b.y2), (10, 20, 30, 40));
    }

    #[test]
    fn test_compute_union_multiple() {
        let boxes = vec![bbox(0, 0, 10, 10), bbox(5, 5, 20, 20), bbox(15, 0, 25, 5)];
        let u = compute_union_bbox(&boxes).unwrap();
        assert_eq!((u.x1, u.y1, u.x2, u.y2), (0, 0, 25, 20));
    }

    #[test]
    fn test_intersects() {
        let a = bbox(0, 0, 10, 10);
        let b = bbox(5, 5, 15, 15);
        let c = bbox(11, 11, 20, 20);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_bbox_dimensions() {
        let b = bbox(5, 10, 25, 40);
        assert_eq!(b.width(), 20);
        assert_eq!(b.height(), 30);
    }

    #[test]
    fn test_closest_box_to_point_inside() {
        let boxes = vec![bbox(0, 0, 10, 10), bbox(50, 50, 60, 60)];
        assert_eq!(closest_box_to_point(&boxes, 5, 5), 0);
        assert_eq!(closest_box_to_point(&boxes, 55, 55), 1);
    }

    #[test]
    fn test_closest_box_to_point_outside() {
        let boxes = vec![bbox(0, 0, 10, 10), bbox(50, 0, 60, 10)];
        // Point at (45, 5): closer to box 1's left edge (dx=5) than box 0's right edge (dx=35).
        assert_eq!(closest_box_to_point(&boxes, 45, 5), 1);
    }

    // ── ONNX-gated tests ───────────────────────────────────────────────────

    #[cfg(feature = "onnx")]
    #[test]
    fn test_merge_overlapping_collapses() {
        let boxes = vec![
            bbox(0, 0, 10, 10),
            bbox(8, 8, 20, 20), // overlaps first
            bbox(50, 50, 60, 60),
        ];
        let merged = merge_overlapping(boxes);
        assert_eq!(merged.len(), 2, "two non-overlapping groups should remain");
        let u = &merged[0];
        assert_eq!((u.x1, u.y1, u.x2, u.y2), (0, 0, 20, 20));
    }

    /// Lens-crop simulation: three separate text regions in Unit-test-sample-texts.png.
    ///
    /// Expected output validated against the dbnet-test prototype with
    /// threshold=0.2, dilation=16, pad=32×32.
    #[cfg(feature = "onnx")]
    #[test]
    fn test_detect_lens_crop_returns_three_boxes() {
        let root = env!("CARGO_MANIFEST_DIR");
        let image_path = format!("{root}/tests/fixtures/Unit-test-sample-texts.png");
        let detector = DbNetDetector::from_bytes(EMBEDDED_MODEL, 0.2, 16, 32, 32)
            .expect("failed to load embedded model");
        let image =
            image::open(&image_path).expect("failed to open Unit-test-sample-texts.png");
        let boxes = detector.detect(&image);
        assert_eq!(
            boxes.len(),
            3,
            "expected 3 text regions, got {}: {boxes:?}",
            boxes.len()
        );
    }

    /// Fullscreen capture simulation: two dialogue regions in OCR-Demo-JP2EN.png.
    ///
    /// Expected output validated against the dbnet-test prototype with
    /// threshold=0.2, dilation=16, pad=32×32: header + dialogue merge into one
    /// box, subtitle is the second.
    #[cfg(feature = "onnx")]
    #[test]
    fn test_detect_fullscreen_returns_two_boxes() {
        let root = env!("CARGO_MANIFEST_DIR");
        let image_path = format!("{root}/tests/fixtures/OCR-Demo-JP2EN.png");
        let detector = DbNetDetector::from_bytes(EMBEDDED_MODEL, 0.2, 16, 32, 32)
            .expect("failed to load embedded model");
        let image = image::open(&image_path).expect("failed to open OCR-Demo-JP2EN.png");
        let boxes = detector.detect(&image);
        assert_eq!(
            boxes.len(),
            2,
            "expected 2 text regions, got {}: {boxes:?}",
            boxes.len()
        );
    }
}

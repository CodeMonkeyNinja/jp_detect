//! Detects text regions in images, draws bounding boxes on a greyscale copy,
//! and writes the annotated PNG and a JSON file to `/dev/shm/jp_detect/`.
//!
//! Usage:
//!   cargo run --example detect_and_draw --features onnx -- <image1> [image2 …]
//!
//! With no arguments the two built-in test fixtures are used.

use std::path::Path;

use image::Rgba;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use jp_detect::build_text_detector;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let root = env!("CARGO_MANIFEST_DIR");
    let paths: Vec<String> = if args.is_empty() {
        vec![
            format!("{root}/tests/fixtures/Unit-test-sample-texts.png"),
            format!("{root}/tests/fixtures/OCR-Demo-JP2EN.png"),
        ]
    } else {
        args
    };

    let out_dir = Path::new("/dev/shm/jp_detect");
    std::fs::create_dir_all(out_dir)?;

    let detector = build_text_detector(None, 0.2, 16, 32, 32)?
        .expect("build_text_detector returned None — was the `onnx` feature enabled?");

    for path in &paths {
        let src = image::open(path)?;

        let boxes = detector.detect(&src);

        // Convert to greyscale then RGBA so we can overlay coloured boxes.
        let mut canvas = src.grayscale().to_rgba8();

        let red = Rgba([255u8, 0, 0, 255]);
        for b in &boxes {
            let w = b.x2.saturating_sub(b.x1);
            let h = b.y2.saturating_sub(b.y1);
            if w == 0 || h == 0 {
                continue;
            }
            // Draw a 2-pixel-thick border by layering two rects.
            for inset in 0u32..2 {
                let rect = Rect::at(
                    (b.x1 + inset) as i32,
                    (b.y1 + inset) as i32,
                )
                .of_size(w.saturating_sub(inset * 2), h.saturating_sub(inset * 2));
                draw_hollow_rect_mut(&mut canvas, rect, red);
            }
        }

        let stem = Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        let img_out = out_dir.join(format!("{stem}_annotated.png"));
        canvas.save(&img_out)?;
        println!("image : {}", img_out.display());

        let json_out = out_dir.join(format!("{stem}_boxes.json"));
        std::fs::write(&json_out, boxes_to_json(&boxes))?;
        println!("json  : {} ({} boxes)", json_out.display(), boxes.len());
        println!();
    }

    Ok(())
}

fn boxes_to_json(boxes: &[jp_detect::TextBoundingBox]) -> String {
    let entries: Vec<String> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            format!(
                "  {{\"index\": {i}, \"x1\": {}, \"y1\": {}, \"x2\": {}, \"y2\": {}, \
                 \"width\": {}, \"height\": {}}}",
                b.x1,
                b.y1,
                b.x2,
                b.y2,
                b.width(),
                b.height()
            )
        })
        .collect();
    format!("[\n{}\n]\n", entries.join(",\n"))
}

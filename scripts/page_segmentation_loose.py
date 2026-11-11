"""Split a manuscript page into line boxes with a horizontal ink profile.

Simple on purpose: works for fairly straight lines with white space between them.
Strongly slanted or touching lines and marginal notes need a trained segmenter.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage


def _paper_bounds(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the main bright sheet and inset past irregular scan edges."""
    height, width = gray.shape
    center = gray[height // 4 : height - height // 4, width // 4 : width - width // 4]
    paper = gray >= 0.72 * np.median(center)
    rows = np.flatnonzero(paper.mean(axis=1) > 0.60)
    cols = np.flatnonzero(paper.mean(axis=0) > 0.60)
    if not rows.size or not cols.size:
        return 0, 0, width, height
    # Apply the inset only at edges where a dark surround was detected.
    xpad, ypad = round(width * .018), round(height * .012)
    left = int(cols[0]) + xpad if cols[0] else 0
    right = int(cols[-1]) + 1 - xpad if cols[-1] < width - 1 else width
    top = int(rows[0]) + ypad if rows[0] else 0
    bottom = int(rows[-1]) + 1 - ypad if rows[-1] < height - 1 else height
    if left >= right or top >= bottom:
        return 0, 0, width, height
    return left, top, right, bottom


def _split_tall(start: int, end: int, profile: np.ndarray, typical: float) -> list[tuple[int, int]]:
    """Cut a band that holds several touching lines at its emptiest rows."""
    count = round((end - start) / typical)
    if count < 2:
        return [(start, end)]
    cuts, reach = [start], max(1, int(typical / 3))
    for i in range(1, count):
        target = start + int(i * (end - start) / count)
        lo, hi = max(cuts[-1] + 1, target - reach), min(end - 1, target + reach)
        cuts.append(lo + int(np.argmin(profile[lo : hi + 1])) if hi > lo else target)
    cuts.append(end)
    return list(zip(cuts[:-1], cuts[1:]))


def _straighten_ink(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align a mildly slanted page for line detection, retaining column offsets."""
    height, width = ink.shape
    ys, xs = np.nonzero(ink)
    columns = np.arange(width) - width // 2
    offsets = np.zeros(width, dtype=int)
    best = 0.0
    for slope in (0, -.0125, .0125, -.025, .025, -.0375, .0375, -.05, .05):
        shifts = np.rint(columns * slope).astype(int)
        rows = ys - shifts[xs]
        profile = np.bincount(rows + width, minlength=height + 2 * width)
        profile = ndimage.gaussian_filter1d(profile.astype(float), sigma=1.5)
        score = np.dot(profile, profile) * (1 - 2 * abs(slope))
        if score > best:
            best, offsets = score, shifts
    rows = ys - offsets[xs]
    inside = (rows >= 0) & (rows < height)
    result = np.zeros_like(ink)
    result[rows[inside], xs[inside]] = True
    return result, offsets


def _trace_polygon(ink, left, top, right, bottom, offsets, line_height):
    """Follow the local ink center and estimate a top and bottom edge."""
    height, _ = ink.shape
    center = (top + bottom - 1) / 2
    reach = max(3, round(line_height * .40))
    candidates = np.arange(max(0, round(center) - reach),
                           min(height, round(center) + reach + 1))
    step = max(5, round(line_height * 1.4))
    xs = np.linspace(left, right - 1, max(2, 1 + (right - left) // step)).astype(int)
    row_min = max(0, top - reach)
    row_max = min(height, bottom + reach)
    profiles = []
    for x in xs:
        stripe = ink[row_min:row_max, max(left, x - step):min(right, x + step + 1)]
        profiles.append(stripe.sum(axis=1))
    profiles = np.asarray(profiles)
    smooth = ndimage.gaussian_filter1d(profiles.astype(float),
                                      sigma=max(1, line_height * .14), axis=1)
    strength = np.maximum(smooth.max(axis=1, keepdims=True), 1)
    emissions = smooth[:, candidates - row_min] / strength
    emissions -= .20 * ((candidates - center) / reach) ** 2

    # Penalize abrupt jumps in y between adjacent windows.
    transition = .28 * ((candidates[:, None] - candidates) / reach) ** 2
    links = []
    scores = emissions[0]
    for emission in emissions[1:]:
        choices = scores[:, None] - transition
        links.append(choices.argmax(axis=0))
        scores = emission + choices.max(axis=0)
    state = int(scores.argmax())
    path = [state]
    for link in reversed(links):
        state = int(link[state])
        path.append(state)
    centers = candidates[path[::-1]]

    upper, lower = [], []
    ys = np.arange(row_min, row_max)
    half = max(3, line_height * .85)
    margin = max(2, round(line_height * .15))
    for cy, profile in zip(centers, profiles):
        weights = profile * (np.abs(ys - cy) <= half)
        if weights.sum() > 0:
            cdf = np.cumsum(weights)
            low, high = ys[np.searchsorted(cdf, cdf[-1] * np.array([.04, .96]))]
        else:
            low, high = cy - half / 2, cy + half / 2
        upper.append(min(low - margin, cy - .90 * line_height))
        lower.append(max(high + margin, cy + .90 * line_height))
    upper = np.clip(ndimage.median_filter(upper, size=3) + offsets[xs], 0, height - 1)
    lower = np.clip(ndimage.median_filter(lower, size=3) + offsets[xs], 0, height - 1)
    return list(zip(xs, upper)) + list(zip(xs[::-1], lower[::-1]))


def find_lines(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Return the deskew variant's rectangular line boxes."""
    return _find_lines(image, trace=False)


def find_line_polygons(image: Image.Image) -> list[list[tuple[int, int]]]:
    """Return tracked line polygons in the input image's pixel coordinates."""
    return _find_lines(image, trace=True)


def _whole_image(width, height, trace):
    if trace:
        return [[(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]]
    return [(0, 0, width, height)]


def _find_lines(image: Image.Image, trace: bool):
    """Line boxes (left, top, right, bottom) in reading order, top to bottom.

    A wide, already cropped single-line image is preserved whole.
    """
    gray = ImageOps.grayscale(image)
    width, height = gray.size
    # Analyse at a consistent scale, but keep the output crops at full resolution.
    scale = min(1.0, 1800 / max(width, height))
    if scale < 1:
        gray = gray.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                           Image.Resampling.LANCZOS)
    w, h = gray.size
    left_edge, top_edge, right_edge, bottom_edge = _paper_bounds(np.asarray(gray))
    paper = gray.crop((left_edge, top_edge, right_edge, bottom_edge))
    pixels = np.asarray(paper, dtype=np.int16)
    background = np.asarray(paper.filter(ImageFilter.GaussianBlur(max(2, w / 125))),
                            dtype=np.int16)
    contrast = background - pixels
    ink = contrast >= max(12, float(np.percentile(contrast, 99.5)) * .28)
    if scale < 1:
        # Thick background, seals, long creases and isolated specks do not
        # contribute to the projection used to locate handwriting.
        labels, count = ndimage.label(ink, structure=np.ones((3, 3)))
        sizes = np.bincount(labels.ravel())
        keep = np.zeros(count + 1, dtype=bool)
        for label, (ys, xs) in enumerate(ndimage.find_objects(labels), 1):
            cw, ch = xs.stop - xs.start, ys.stop - ys.start
            keep[label] = (sizes[label] >= max(3, w // 350)
                           and cw <= w * .16 and ch <= h * .04
                           and sizes[label] <= w * h * .001
                           and cw <= 9 * ch and ch <= 13 * cw)
        ink = keep[labels]
        # A text line has clusters of strokes at nearby x positions.
        density = ndimage.uniform_filter(ink.astype(np.float32),
                                        size=(max(3, h // 120), max(7, w // 25)))
        ink &= density > .11
    ink[:, ink.mean(axis=0) > .42] = False
    ink[ink.mean(axis=1) > .72, :] = False
    paper_h, paper_w = ink.shape
    offsets = np.zeros(paper_w, dtype=int)
    if scale < 1:
        ink, offsets = _straighten_ink(ink)
    window = max(1, h // 230)
    profile = np.convolve(ink.sum(axis=1), np.ones(window) / window, mode="same")
    if profile.max() == 0:
        return []
    rows = profile > max(2, .004 * paper_w, .22 * np.percentile(profile[profile > 0], 95))

    bands, start = [], None
    for y, on in enumerate(np.append(rows, False)):
        if on and start is None:
            start = y
        elif not on and start is not None:
            bands.append((start, y))
            start = None
    if not bands:
        return []
    if scale == 1 and len(bands) <= 2 and bands[-1][1] - bands[0][0] < h / 6:
        return _whole_image(width, height, trace)
    typical = float(np.median([b - a for a, b in bands]))
    bands = [(a, b) for a, b in bands if b - a >= max(2, .35 * typical)]
    if not bands:
        return []
    typical = float(np.median([b - a for a, b in bands]))
    bands = [part for a, b in bands for part in _split_tall(a, b, profile, typical)]
    if len(bands) == 1 and width >= 3 * height:
        return _whole_image(width, height, trace)

    pad = max(2, int(.40 * typical))
    centers = [(top + bottom) / 2 for top, bottom in bands]
    boxes = []
    for i, (top, bottom) in enumerate(bands):
        # Limit added padding when the next detected line is close.
        center = centers[i]
        top_limit, bottom_limit = 0, paper_h
        if i:
            top_limit = min(top, round(center - .56 * (center - centers[i - 1])))
        if i + 1 < len(bands):
            bottom_limit = max(bottom, round(center + .56 * (centers[i + 1] - center)))
        top = max(0, top_limit, top - pad)
        bottom = min(paper_h, bottom_limit, bottom + pad)
        cols = np.flatnonzero(ink[top:bottom].sum(axis=0) >= 2)
        if not cols.size:
            continue
        left, right = np.percentile(cols, [1, 99])
        left = max(0, int(left) - 2 * pad)
        right = min(paper_w, int(right) + 1 + 2 * pad)
        if trace:
            polygon = _trace_polygon(ink, left, top, right, bottom, offsets, typical)
            boxes.append([
                (min(width - 1, round((left_edge + x) * width / w)),
                 min(height - 1, round((top_edge + y) * height / h)))
                for x, y in polygon
            ])
            continue
        line_offsets = offsets[left:right]
        top = max(0, top + int(line_offsets.min()))
        bottom = min(paper_h, bottom + int(line_offsets.max()))
        boxes.append((
            (left_edge + left) * width // w,
            (top_edge + top) * height // h,
            ((left_edge + right) * width + w - 1) // w,
            ((top_edge + bottom) * height + h - 1) // h,
        ))
    return boxes


def split_page(image: Image.Image) -> list[tuple[tuple[int, int, int, int], Image.Image]]:
    """Return (box, RGB crop) pairs in top-to-bottom order.

    Coordinates refer to the EXIF-oriented page; right/bottom are exclusive.
    Transparent pixels are composited onto white before finding ink.
    """
    page = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", page.size, "white")
    page = Image.alpha_composite(background, page).convert("RGB")
    return [(box, page.crop(box)) for box in find_lines(page)]


def _polygon_crop(page, polygon):
    xs, ys = zip(*polygon)
    left, top, right, bottom = min(xs), min(ys), max(xs) + 1, max(ys) + 1
    box = (left, top, right, bottom)
    crop = page.crop(box)
    mask = Image.new("L", crop.size)
    ImageDraw.Draw(mask).polygon([(x - left, y - top) for x, y in polygon], fill=255)
    return box, Image.composite(crop, Image.new("RGB", crop.size, "white"), mask)


def export_page(source, output_dir, *, contours=True):
    """Export PNG lines, manifest.json and preview.png into a new directory.

    No recognition model is loaded. Existing destinations are never overwritten.
    """
    import json
    from pathlib import Path

    source, output_dir = Path(source), Path(output_dir)
    with Image.open(source) as image:
        oriented = ImageOps.exif_transpose(image).convert("RGBA")
        page = Image.alpha_composite(Image.new("RGBA", oriented.size, "white"), oriented).convert("RGB")
    if contours:
        lines = [(*_polygon_crop(page, polygon), polygon) for polygon in find_line_polygons(page)]
    else:
        lines = [(box, crop, None) for box, crop in split_page(page)]
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    preview = page.copy()
    draw = ImageDraw.Draw(preview)
    for number, (box, crop, polygon) in enumerate(lines, 1):
        filename = f"line_{number:04d}.png"
        crop.save(output_dir / filename)
        records.append({"line_id": f"line_{number:04d}", "image": filename, "bbox": list(box)})
        left, top, right, bottom = box
        if polygon is None:
            draw.rectangle((left, top, right - 1, bottom - 1), outline="red", width=2)
        else:
            records[-1]["polygon"] = [list(point) for point in polygon]
            draw.line(polygon + [polygon[0]], fill="red", width=2)
        draw.text((left, top), str(number), fill="blue")
    preview.save(output_dir / "preview.png")
    manifest = {"source": str(source.resolve()), "page_size": list(page.size),
                "coordinate_system": "EXIF-oriented pixels; left, top, right, bottom; right/bottom exclusive",
                "reading_order": "top-to-bottom, single column", "lines": records}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Export ordered line images using the deskew variant.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contours", action=argparse.BooleanOptionalAction, default=True,
                        help="Trace and mask line polygons (default). Use --no-contours for rectangles.")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Missing image: {args.image}")
    if args.output_dir.exists():
        parser.error("Choose a new output directory.")
    result = export_page(args.image, args.output_dir, contours=args.contours)
    print(f"Exported {len(result['lines'])} lines to {args.output_dir}")

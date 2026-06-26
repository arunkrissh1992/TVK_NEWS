"""Generate src/tnmi/tn_map.py from a Tamil Nadu districts GeoJSON.

Converts real district boundary geometry into compact SVG paths so the
dashboard renders a true, clickable Tamil Nadu map with zero runtime
dependencies. Run once whenever the source geometry changes:

    .venv/bin/python scripts/build_tn_map.py /tmp/tn-districts-hd.geojson

Source data: https://github.com/datta07/INDIAN-SHAPEFILES (38 districts,
post-2020 boundaries, including Mayiladuthurai).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tnmi.districts import canonical_district  # noqa: E402

VIEW_W = 640.0
SIMPLIFY_TOLERANCE = 0.9  # px in SVG space — keeps shapes crisp but compact
MIN_RING_AREA = 6.0  # px² — drop tiny islets that add bytes, not meaning


def _perp_distance(point, start, end) -> float:
    (px, py), (sx, sy), (ex, ey) = point, start, end
    dx, dy = ex - sx, ey - sy
    if dx == dy == 0:
        return math.hypot(px - sx, py - sy)
    t = ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = sx + t * dx, sy + t * dy
    return math.hypot(px - cx, py - cy)


def douglas_peucker(points, tolerance):
    if len(points) < 3:
        return points
    # Iterative stack version — rings can have thousands of points.
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        max_dist, index = 0.0, 0
        for i in range(first + 1, last):
            dist = _perp_distance(points[i], points[first], points[last])
            if dist > max_dist:
                max_dist, index = dist, i
        if max_dist > tolerance:
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))
    return [p for p, k in zip(points, keep) if k]


def ring_area(points) -> float:
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/tn-districts-hd.geojson")
    data = json.loads(source.read_text(encoding="utf-8"))
    features = data["features"]

    # Collect every coordinate to compute the projection bounds.
    lons, lats = [], []

    def walk(coords):
        if isinstance(coords[0], (int, float)):
            lons.append(coords[0])
            lats.append(coords[1])
        else:
            for c in coords:
                walk(c)

    for feature in features:
        walk(feature["geometry"]["coordinates"])

    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    mid_lat = math.radians((min_lat + max_lat) / 2)
    # Equirectangular projection with latitude correction; y axis flipped.
    x_span = (max_lon - min_lon) * math.cos(mid_lat)
    y_span = max_lat - min_lat
    scale = VIEW_W / x_span
    view_h = y_span * scale

    def project(lon, lat):
        x = (lon - min_lon) * math.cos(mid_lat) * scale
        y = (max_lat - lat) * scale
        return x, y

    paths: dict[str, str] = {}
    labels: dict[str, tuple[float, float]] = {}

    for feature in features:
        raw_name = feature["properties"]["dtname"]
        name = canonical_district(raw_name)
        if name is None:
            raise SystemExit(f"District name {raw_name!r} did not resolve to a canonical district")
        geometry = feature["geometry"]
        polygons = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )

        d_parts: list[str] = []
        best_ring, best_area = None, -1.0
        for polygon in polygons:
            for ring_index, ring in enumerate(polygon):
                pts = [project(lon, lat) for lon, lat in ring]
                pts = douglas_peucker(pts, SIMPLIFY_TOLERANCE)
                if len(pts) < 4:
                    continue
                area = ring_area(pts)
                if area < MIN_RING_AREA and ring_index == 0 and len(polygons) > 1:
                    continue  # tiny islet
                if ring_index == 0 and area > best_area:
                    best_area, best_ring = area, pts
                coords = " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                d_parts.append(f"M{coords}Z")
        paths[name] = " ".join(d_parts)
        if best_ring:
            cx = sum(x for x, _ in best_ring) / len(best_ring)
            cy = sum(y for _, y in best_ring) / len(best_ring)
            labels[name] = (round(cx, 1), round(cy, 1))

    assert len(paths) == 38, f"expected 38 districts, got {len(paths)}"

    out = PROJECT_ROOT / "src" / "tnmi" / "tn_map.py"
    body_paths = ",\n".join(f"    {name!r}: {d!r}" for name, d in sorted(paths.items()))
    body_labels = ",\n".join(f"    {name!r}: {pt!r}" for name, pt in sorted(labels.items()))
    out.write_text(
        '"""Real Tamil Nadu district boundaries as SVG paths.\n\n'
        "GENERATED by scripts/build_tn_map.py — do not edit by hand.\n"
        "Source: datta07/INDIAN-SHAPEFILES (38 districts, post-2020 boundaries),\n"
        "equirectangular projection, Douglas-Peucker simplified.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        f"TN_MAP_VIEWBOX = \"0 0 {VIEW_W:.0f} {view_h:.0f}\"\n\n"
        "# canonical district name -> SVG path data\n"
        "TN_MAP_PATHS: dict[str, str] = {\n" + body_paths + ",\n}\n\n"
        "# canonical district name -> (x, y) label anchor (largest-ring centroid)\n"
        "TN_MAP_LABELS: dict[str, tuple[float, float]] = {\n" + body_labels + ",\n}\n",
        encoding="utf-8",
    )
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({size_kb:.0f} KB, viewBox 0 0 {VIEW_W:.0f} {view_h:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

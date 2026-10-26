"""The defense deck is generated from real traces and the checked-in evidence."""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_slides", ROOT / "scripts/build_slides.py")
build_slides = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_slides)


def test_deck_has_rubric_length_real_ecgs_and_evidence_numbers():
    html = build_slides.build_html()
    assert 8 <= html.count('<section class="slide') <= 12
    # Five superclass traces, the 12-lead record, the cover and the annotated beat.
    assert len(re.findall(r'class="ecg"', html)) == 5 + 12 + 1 + 1
    for number in ("0,692", "0,744", "0,750", "0,039"):
        assert number in html
    assert "BORRADOR" not in html.upper()


def test_trace_svg_scales_to_requested_box():
    svg = build_slides.trace_svg([0.0, 1.0, -1.0, 0.0] * 100, 4, 400, 80)
    assert 'viewBox="0 0 400 80"' in svg
    xs = [float(p.split(",")[0]) for p in re.search(r'points="([^"]+)"', svg).group(1).split()]
    assert xs[0] == 0.0 and abs(xs[-1] - 400) < 0.2

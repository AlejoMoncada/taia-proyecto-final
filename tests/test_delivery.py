"""Run with unittest in the PDF tools environment; no ML dependencies required."""
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF_TOOLS = all(importlib.util.find_spec(name) for name in ("reportlab", "pypdf"))


@unittest.skipUnless(shutil.which("git"), "Git is needed to check binary artifact attributes")
class GitArtifactTests(unittest.TestCase):
    def test_pdf_and_png_have_no_text_conversion_or_text_diff(self):
        result = subprocess.run(["git", "check-attr", "-z", "text", "diff", "--",
                                 "output/pdf/informe.pdf", "docs/evidence/curves.png"],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        values = result.stdout.rstrip("\0").split("\0")
        for path, attribute, value in zip(values[::3], values[1::3], values[2::3]):
            self.assertEqual(value, "unset", f"{path}: {attribute} must be unset for binary artifacts")


@unittest.skipUnless(PDF_TOOLS, "PDF checks need reportlab and pypdf in a separate tools environment")
class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="taia-delivery-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        (self.root / "scripts").mkdir()
        shutil.copyfile(ROOT / "scripts/build_delivery.py", self.root / "scripts/build_delivery.py")
        shutil.copytree(ROOT / "docs/evidence", self.root / "docs/evidence")
        shutil.copytree(ROOT / "assets/brand", self.root / "assets/brand")
        if (ROOT / "assets/fonts").exists():
            shutil.copytree(ROOT / "assets/fonts", self.root / "assets/fonts")
        for name in ("plan.md", "informe.md"):
            shutil.copyfile(ROOT / "docs" / name, self.root / "docs" / name)
        # The deck comes from scripts/build_slides.py; the build only checks it.
        (self.root / "output/pdf").mkdir(parents=True)
        shutil.copyfile(ROOT / "output/pdf/diapositivas.pdf", self.root / "output/pdf/diapositivas.pdf")
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(self.root / "scripts/build_delivery.py"), *args],
                              cwd=self.root, capture_output=True, text=True)

    def output_hashes(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (self.root / "output/pdf").glob("*.pdf")}

    def test_build_matches_rubric_and_preserves_draft_status(self):
        from pypdf import PdfReader

        for name, count in {"plan": 5, "informe": 10, "ejecucion": 6, "diapositivas": 11}.items():
            reader = PdfReader(self.root / "output/pdf" / f"{name}.pdf")
            self.assertEqual(len(reader.pages), count)
            self.assertFalse(any("BORRADOR" in page.extract_text().upper() for page in reader.pages))
        self.assertIn("0,750", "".join(p.extract_text() for p in PdfReader(self.root / "output/pdf/informe.pdf").pages))

    def test_check_is_read_only(self):
        before = self.output_hashes()
        result = self.run_cli("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output_hashes(), before)

    def test_incomplete_model_table_is_rejected_before_any_pdf_changes(self):
        before = self.output_hashes()
        path = self.root / "docs/evidence/table_test.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows[:-1])
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["snapshot_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0, "A missing model must not silently produce a partial table")
        self.assertEqual(self.output_hashes(), before)


if __name__ == "__main__":
    unittest.main()

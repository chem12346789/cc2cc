import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from scripts.clean_validation_csv import clean_csv, main


class TestCleanValidationCsv(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "def2-QZVP(D)_atom-123"
        self.folder.mkdir()
        self.path = self.folder / "data.csv"
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def test_default_dry_run_and_apply_preserve_retained_bytes(self):
        data = (b'\xef\xbb\xbfname,scf_d3bj_ene,note\r\n'
                b'empty,,remove\r\nspace,  ,remove\r\n'
                b'"quoted,\r\nname",-1.200,"keep, exactly"\r\n'
                b'zero,0,keep\r\nnan,NaN,keep')
        self.path.write_bytes(data)
        main(["--root", str(self.root)])
        self.assertEqual(self.path.read_bytes(), data)
        self.assertIn("2 rows would be removed", self.output.getvalue())
        self.assertIn("name='empty'", self.output.getvalue())
        main(["--root", str(self.root), "--apply"])
        self.assertEqual(
            self.path.read_bytes(),
            data.replace(b'empty,,remove\r\nspace,  ,remove\r\n', b''),
        )
        self.assertEqual(clean_csv(self.path, apply=True), 0)
        self.assertEqual(list(self.folder.iterdir()), [self.path])

    def test_recursive_scope_and_symlinks(self):
        nested = self.folder / "nested"
        nested.mkdir()
        target = nested / "data.csv"
        target.write_text("name,scf_d3bj_ene\nbad,\n")
        outside = self.root / "unrelated.csv"
        outside.write_bytes(target.read_bytes())
        (self.folder / "linked.csv").symlink_to(outside)
        main(["--root", str(self.root), "--apply"])
        self.assertEqual(target.read_text(), "name,scf_d3bj_ene\n")
        self.assertEqual(outside.read_text(), "name,scf_d3bj_ene\nbad,\n")
        self.assertIn("scanned 1 CSV files", self.output.getvalue())

    def test_missing_column_is_skipped(self):
        self.path.write_text("name,other\nkeep,\n")
        self.assertEqual(clean_csv(self.path, apply=True), 0)
        self.assertIn("SKIP", self.output.getvalue())
        self.assertEqual(self.path.read_text(), "name,other\nkeep,\n")

    def test_malformed_csv_is_not_rewritten(self):
        for text in ("name,scf_d3bj_ene\nbad,\nshort\n",
                     "name,scf_d3bj_ene,scf_d3bj_ene\nbad,,\n"):
            self.path.write_text(text)
            with self.assertRaises(ValueError):
                clean_csv(self.path, apply=True)
            self.assertEqual(self.path.read_text(), text)

    def test_empty_result_keeps_header_and_permissions(self):
        self.path.write_text("name,scf_d3bj_ene\nbad,\n")
        self.path.chmod(0o640)
        self.assertEqual(clean_csv(self.path, apply=True), 1)
        self.assertEqual(self.path.read_text(), "name,scf_d3bj_ene\n")
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o640)


if __name__ == "__main__":
    unittest.main()

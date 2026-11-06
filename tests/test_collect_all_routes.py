import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_all_routes import discover_routes, nonempty_scene_dirs  # noqa: E402


class CollectAllRoutesTest(unittest.TestCase):
    def test_discovers_routes_in_filename_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            routes_dir = config_dir / "routes"
            routes_dir.mkdir()
            (routes_dir / "Town02.yaml").write_text(
                "route:\n  name: Town02\n", encoding="utf-8"
            )
            (routes_dir / "Town01.yaml").write_text(
                "route:\n  name: Town01\n", encoding="utf-8"
            )

            entries = discover_routes(routes_dir, config_dir)

            self.assertEqual([entry["name"] for entry in entries], ["Town01", "Town02"])
            self.assertEqual(entries[0]["route_file"], "routes/Town01.yaml")

    def test_reports_only_nonempty_scene_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            (output_root / "Town01").mkdir()
            (output_root / "Town02").mkdir()
            (output_root / "Town02" / "metadata.json").write_text("{}", encoding="utf-8")
            entries = [{"name": "Town01"}, {"name": "Town02"}]

            conflicts = nonempty_scene_dirs(output_root, entries)

            self.assertEqual(conflicts, [output_root / "Town02"])


if __name__ == "__main__":
    unittest.main()

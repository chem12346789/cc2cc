import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from unittest import TestCase


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/calculator/calculator_diet100.bash"


class TestCalculatorDiet100(TestCase):
    def test_energy_column_forwarding_and_output(self):
        source = SCRIPT.read_text()
        for custom in (False, True):
            with self.subTest(custom=custom), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "log").mkdir()
                bin_dir = root / "bin"
                bin_dir.mkdir()
                for command in ("find", "rm"):
                    stub = bin_dir / command
                    stub.write_text("#!/bin/bash\nexit 0\n")
                    stub.chmod(0o755)
                if custom:
                    (bin_dir / "find").write_text(
                        "#!/bin/bash\n"
                        'if [[ "$1" == validate/* ]]; then\n'
                        '    printf "find: %s: No such file or directory\\n" "$1" >&2\n'
                        '    exit 1\n'
                        'fi\n'
                    )
                python = root / "anaconda3/envs/pyscf/bin/python"
                python.parent.mkdir(parents=True)
                python.write_text(
                    "#!/bin/bash\n"
                    'printf "%s\\n" "$@" >> "$ARG_LOG"\n'
                    'printf "%s\\n" "---" >> "$ARG_LOG"\n'
                    'printf "summary 1.0 2.0 DONE\\n"\n'
                    'if [[ " $* " != *" --epoch 1000 "* ]]; then\n'
                    '    printf "Converged tests: 1/1\\n"\n'
                    'fi\n'
                )
                python.chmod(0o755)
                script = source
                if custom:
                    script = script.replace(
                        '"atom-dft:1000:scf_d3bj_ene"',
                        '"atom-dft:1000:scf_d3bj_ene"\n'
                        '    "atom-dft:1000:post_scf_vv10_ene"\n'
                        '    "atom-dft:1000:other_ene"',
                        1,
                    )
                expected_pairs = re.findall(
                    r'"([^"]+:[^"]+:[^"]+)"',
                    script.split("export model_load_epoch_pairs=(", 1)[1].split(")", 1)[0],
                )
                arg_log = root / "args"
                result = subprocess.run(
                    ["bash"], input=script, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=root, check=True,
                    env={
                        **os.environ,
                        "HOME": str(root),
                        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                        "ARG_LOG": str(arg_log),
                    },
                )
                calls = [
                    call.splitlines()
                    for call in arg_log.read_text().split("---\n") if call
                ]
                self.assertEqual(len(calls), len(expected_pairs))
                for pair, call in zip(expected_pairs, calls):
                    model, epoch, energy_column = pair.split(":")
                    self.assertEqual(call[call.index("--model_load") + 1], model)
                    self.assertEqual(call[call.index("--epoch") + 1], epoch)
                    self.assertEqual(call[call.index("--energy_column") + 1], energy_column)
                    self.assertEqual(call[call.index("--basis") + 1], "def2-QZVP(D)")
                output_lines = result.stdout.splitlines()
                table_lines = output_lines[:len(expected_pairs) + 2]
                expected_errors = []
                if custom:
                    for pair in expected_pairs:
                        model, epoch, _ = pair.split(":")
                        expected_errors.append(
                            f"find: validate/def2-QZVP(D)_{model}/{epoch}: "
                            "No such file or directory"
                        )
                self.assertEqual(output_lines[len(table_lines):], expected_errors)
                header = table_lines[0]
                pipe_positions = [i for i, char in enumerate(header) if char == "|"]
                self.assertEqual(len(pipe_positions), 8)
                for line in table_lines:
                    self.assertNotIn("\t", line)
                    self.assertEqual(
                        [i for i, char in enumerate(line) if char == "|"],
                        pipe_positions,
                    )
                self.assertEqual([cell.strip() for cell in header.split("|")[1:-1]], [
                    "Model", "Epoch", "AI", "Corrected", "Processed",
                    "Converged", "Energy_column",
                ])
                self.assertNotIn("Basis", header)
                separator = table_lines[1]
                self.assertEqual(set(separator), {"|", "-"})
                rows = table_lines[2:]
                self.assertEqual(len(rows), len(expected_pairs))
                for pair, row in zip(expected_pairs, rows):
                    model, epoch, energy_column = pair.split(":")
                    values = [cell.strip() for cell in row.split("|")[1:-1]]
                    self.assertEqual(len(values), 7)
                    self.assertEqual(values[:2], [model, epoch])
                    self.assertEqual(values[2:4], ["1.00", "2.00"])
                    self.assertEqual(values[4], "DONE")
                    self.assertEqual(values[5], "" if epoch == "1000" else "1/1")
                    energy_label = {
                        "scf_d3bj_ene": "d3bj",
                        "post_scf_vv10_ene": "post_vv10",
                    }.get(energy_column, energy_column)
                    self.assertEqual(values[-1], energy_label)
                self.assertIn("Corrected", result.stdout)
                self.assertIn("d3bj", result.stdout)
                self.assertTrue((root / "log/collect_info_atom-dft_1000_scf_d3bj_ene_cal_100.log").exists())
                if custom:
                    self.assertIn("post_vv10", result.stdout)
                    self.assertIn("other_ene", result.stdout)
                    self.assertTrue((root / "log/collect_info_atom-dft_1000_post_scf_vv10_ene_cal_100.log").exists())

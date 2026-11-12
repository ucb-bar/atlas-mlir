"""Exactness and physical broadcast checks for the vector restoration prototype."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class Power2RestorationPrototypeTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL') == '1', 'select unchanged model explicitly')
    def test_selected_model_exhaustive_normal_products_and_broadcasts(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / 'receipt.json'
            result = subprocess.run([os.environ['IMPORT_MODEL_PYTHON'],
                str(Path(__file__).with_name('power2_restoration_model.py')), str(receipt), str(root)],
                capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            record = json.loads(receipt.read_text())
            self.assertEqual(record['raw_codes'], 15105)
            self.assertEqual(record['intermediate_codes'], 39937)
            self.assertEqual(record['first_multiply_exact_cases'], 1480290)
            self.assertEqual(record['second_multiply_exact_cases'], 3913826)
            self.assertGreater(record['naive_single_column_broadcast_mismatches'], 0)
            self.assertEqual(record['scalar_broadcast_exact_codes'], 98*1024)
            output = os.environ.get('ATLAS_POWER2_RESTORATION_RECEIPT')
            if output:
                Path(output).write_bytes(receipt.read_bytes())


if __name__ == '__main__':
    unittest.main()

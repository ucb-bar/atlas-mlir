"""Extracted-region provenance is rebound while numerical source fields stay exact."""
from dataclasses import replace
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from atlas_mlir_native.options import codegen_options
import test_resident_loop_image as fixture


class ResidentLoopProvenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get('ATLAS_RESIDENT_IMAGE_RECEIPTS')
        if directory:
            with patch.dict(os.environ, ATLAS_RESIDENT_IMAGE_RECEIPTS=str(Path(directory)/'provenance')):
                fixture.ResidentLoopImageTest.setUpClass.__func__(cls)
        else:
            fixture.ResidentLoopImageTest.setUpClass.__func__(cls)

    tearDownClass = classmethod(fixture.ResidentLoopImageTest.tearDownClass.__func__)

    def test_source_provenance_changes_bind_but_calibration_changes_refuse(self):
        source = replace(self.source, source_identity='current extracted region source identity')
        allocation = self.selected.resident_bindings[self.source.digest()]
        selected = replace(self.selected, resident_bindings={source.digest(): allocation})
        with codegen_options(selected):
            bound = self.image.bind(source)
            self.image.verify_binding(source, bound)
        self.assertEqual(bound.receipt['source_digest'], source.digest())
        self.assertNotEqual(bound.receipt['source_digest'], self.source.digest())
        self.assertEqual(bound.binary, self.template)
        calibrated = replace(source, nodes=(*source.nodes[:-1], replace(source.nodes[-1],
            attrs=(*source.nodes[-1].attrs, ('calibration_id', 'changed source calibration')))))
        with codegen_options(replace(selected, resident_bindings={calibrated.digest(): allocation})):
            with self.assertRaisesRegex(ValueError, 'source structure differs'):
                self.image.bind(calibrated)


if __name__ == '__main__':
    unittest.main()

"""Rehashed native files cannot replace source-to-machine template replay."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir_native.options import codegen_options
from atlas_mlir_native.resident_loop_image import ResidentLoopImage
import test_resident_loop_image as fixture


def write_record(path, record):
    path.write_text(json.dumps(record, sort_keys=True, indent=2)+'\n')


def rehash_native(root):
    """Give the attacker every claimed hash, including the pipeline output."""
    pipeline = json.loads((root/'compilation-pipeline.json').read_bytes())
    pipeline['output_sha256'] = sha256((root/'atlas-machine.mlir').read_bytes()).hexdigest()
    write_record(root/'compilation-pipeline.json', pipeline)
    manifest = json.loads((root/'manifest.json').read_bytes())
    manifest['binary_sha256'] = sha256((root/'program.bin').read_bytes()).hexdigest()
    for relative in manifest['stage_artifacts']:
        manifest['stage_artifacts'][relative] = sha256((root/relative).read_bytes()).hexdigest()
    write_record(root/'manifest.json', manifest)


class ResidentLoopTemplateReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get('ATLAS_RESIDENT_IMAGE_RECEIPTS')
        if directory:
            with patch.dict(os.environ, ATLAS_RESIDENT_IMAGE_RECEIPTS=str(Path(directory)/'template-replay')):
                fixture.ResidentLoopImageTest.setUpClass.__func__(cls)
        else:
            fixture.ResidentLoopImageTest.setUpClass.__func__(cls)

    tearDownClass = classmethod(fixture.ResidentLoopImageTest.tearDownClass.__func__)

    def test_authentic_replay_preserves_every_image_and_C_descriptor_byte(self):
        with codegen_options(self.selected):
            expected = self.image.emit_c('checked_image')
            image = ResidentLoopImage(self.source, self.compiled, target_source=fixture.ROOT)
            self.assertEqual(image.emit_c('checked_image'), expected)
            self.assertEqual(image.bind(self.source).binary, self.template)

    def test_rehashed_words_physical_metadata_machine_and_bounds_all_refuse(self):
        results = []
        for tamper in ('word', 'physical_metadata', 'machine_and_words', 'bound'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory(prefix='resident-replay-forgery-') as directory:
                root = Path(directory)/'compiled'
                shutil.copytree(self.compiled, root)
                program = json.loads((root/'physical-program.json').read_bytes())
                if tamper == 'word':
                    index = next(i for i, row in enumerate(program['instructions'])
                                 if row['operation'] not in {'atlas.upper', 'atlas.alu_imm'})
                    row = program['instructions'][index]
                    row['word_u32'] ^= 0x80
                    row['word_hex'] = f"{row['word_u32']:08x}"
                elif tamper == 'physical_metadata':
                    self.assertEqual(program['instructions'][0]['operation'], 'atlas.dma_config')
                    program['instructions'][0]['fields']['base_reg'] = 1
                elif tamper == 'machine_and_words':
                    machine = (root/'atlas-machine.mlir').read_text()
                    # Change the BF16-one splat AND its claimed virtual contract.
                    # The real native emitter accepts this internally consistent
                    # machine, while the trusted source still requires BF16 one.
                    self.assertEqual(machine.count('16256 : i32'), 2)
                    machine = machine.replace('16256 : i32', '0 : i32')
                    (root/'atlas-machine.mlir').write_text(machine)
                    tools = Path(os.environ.get('ATLAS_OOT_BIN_DIR', fixture.ROOT/'build/bin'))
                    emitted = subprocess.run([str(tools/'atlas-emit'), '--program-json', '-'],
                        input=machine, text=True, capture_output=True, check=True)
                    program = json.loads(emitted.stdout)
                else:
                    bound = json.loads((root/'restoration-instruction-bound.json').read_bytes())
                    bound['instruction_bound'] -= 1
                    write_record(root/'restoration-instruction-bound.json', bound)
                write_record(root/'physical-program.json', program)
                binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in program['instructions'])
                (root/'program.bin').write_bytes(binary)
                rehash_native(root)
                self.assertEqual((root/'source-request.json').read_bytes(), (self.compiled/'source-request.json').read_bytes())
                self.assertEqual((root/'atlas-virtual.mlir').read_bytes(), (self.compiled/'atlas-virtual.mlir').read_bytes())
                with codegen_options(self.selected), self.assertRaisesRegex(ValueError, 'trusted source replay'):
                    ResidentLoopImage(self.source, root, target_source=fixture.ROOT)
                results.append(dict(tamper=tamper, rehashed_native_publication=True,
                                    trusted_source_unchanged=True, refused=True))
        (self.root/'template-replay-refusals.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()

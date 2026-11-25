"""Opt-in adjacent device groups retain checked smaller choices."""
from pathlib import Path
import unittest
from unittest.mock import patch


class AdjacentDeviceAdapterTest(unittest.TestCase):
    def test_enlargement_requires_wrapper_and_preserves_smaller_choices(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
        from test_pointwise_narrowing import adapter as base_adapter, case

        self.assertFalse(hasattr(AtlasApplicationAdapter, 'application_region_alternatives'))
        source, base = case(), base_adapter()
        adapter = AdjacentDeviceAdapter(base)
        choices = adapter.application_region_alternatives(source)
        original = tuple(base.plan(source).selected)
        self.assertEqual(tuple(row for row in choices if len(row.nodes) == 1), original)
        self.assertEqual([dict(row.metadata)['kind'] for row in adapter.plan(source).selected],
                         ['pointwise_narrowing'])
        self.assertEqual(adapter.source_exp_policy, base.source_exp_policy)
        self.assertFalse(hasattr(base, '_pointwise_narrowing_groups'))

    def test_table_dispatch_compiles_exports_and_delegates_smaller_choice(self):
        from atlas_mlir_native import pointwise_narrowing
        from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
        from merlin.compiler.partition.semantic_regions import extract_semantic_region
        from test_pointwise_narrowing import adapter as base_adapter, case
        from unittest.mock import Mock

        source, base = case(), base_adapter()
        base.compile_region, base.export_region = Mock(), Mock()
        adapter = AdjacentDeviceAdapter(base)
        choices = adapter.application_region_alternatives(source)
        grouped = next(row for row in choices if dict(row.metadata)['kind'] == 'pointwise_narrowing')
        region = extract_semantic_region(source, ('y', 'out'))
        with patch.object(pointwise_narrowing, 'compile_group') as compile_, \
                patch.object(pointwise_narrowing, 'export_group') as export_:
            adapter.compile_region(None, region, grouped, 'group', Path('segments'), Path('.'), None)
            compile_.assert_called_once()
            self.assertEqual(compile_.call_args.kwargs['destination'], Path('segments/group'))
            self.assertEqual(export_.call_count, 1)
            adapter.export_region(region, grouped, 'group', Path('segments'), Path('.'))
            self.assertEqual(export_.call_count, 2)
        single = next(row for row in choices if len(row.nodes) == 1)
        adapter.compile_region(None, source, single, 'single', Path('segments'), Path('.'), None)
        base.compile_region.assert_called_once()
        adapter.export_region(source, single, 'single', Path('segments'), Path('.'))
        base.export_region.assert_called_once()


if __name__ == "__main__":
    unittest.main()

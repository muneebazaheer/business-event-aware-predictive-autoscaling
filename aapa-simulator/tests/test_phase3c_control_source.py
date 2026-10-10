"""Structural regression test; does not require SimPy.

The runtime experiment ALSO asserts equivalence on every scenario and seed.
"""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'examples' / 'phase3c_compare.py'

class ControlSourceTests(unittest.TestCase):
    def test_event_disabled_is_exact_hpa_factory(self):
        tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_policies')
        dictionary = next(n.value for n in ast.walk(method) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'factories' for t in n.targets))
        entries = {k.value: v for k, v in zip(dictionary.keys, dictionary.values)}
        self.assertEqual(ast.dump(entries['hpa']), ast.dump(entries['event_disabled']))

    def test_runtime_equivalence_assertion_is_present(self):
        src = SCRIPT.read_text(encoding='utf-8')
        self.assertIn('Negative control mismatch', src)
        self.assertIn("'eval_pod_minutes'", src)

if __name__ == '__main__':
    unittest.main()

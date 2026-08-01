import importlib.util
from pathlib import Path

main_path = Path('main.py')
spec = importlib.util.spec_from_file_location('main_under_test', main_path)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print('import_ok')
    print(mod.ValidationAgent2.__module__)
    print(mod._validation_agent2_path)
except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))
    raise

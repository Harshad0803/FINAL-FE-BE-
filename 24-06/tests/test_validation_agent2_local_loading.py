from pathlib import Path
import importlib.util


def test_main_uses_local_validation_agent2_module():
    backend_dir = Path(__file__).resolve().parents[1]
    main_path = backend_dir / "main.py"

    spec = importlib.util.spec_from_file_location("main_under_test", main_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.ValidationAgent2.__module__ == "validation_agent2"
    assert module._validation_agent2_path == backend_dir / "validation_agent2.py"

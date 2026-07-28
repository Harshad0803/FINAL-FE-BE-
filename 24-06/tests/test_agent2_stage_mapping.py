from pathlib import Path
import importlib.util


def _load_agent2_module():
    repo_root = Path(__file__).resolve().parents[1] / ".." / "Credit-Risk-Poc-main"
    path = repo_root / "agent2.py"
    import sys
    sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("agent2_trace", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage_normalization_matches_legacy_and_current_names():
    module = _load_agent2_module()
    agent = module.Agent2(str(Path(__file__).resolve().parents[1] / "../Credit-Risk-Poc-main/rag_store/val_mdd_rules.json"))

    assert agent._normalize_stage("data") == "data_validation"
    assert agent._normalize_stage("data_validation") == "data_validation"
    assert agent._normalize_stage("conceptual") == "conceptual_soundness"
    assert agent._normalize_stage("conceptual_soundness") == "conceptual_soundness"

    assert agent._stages_match("data", "data_validation")
    assert agent._stages_match("data_validation", "data_validation")
    assert agent._stages_match("conceptual", "conceptual_soundness")
    assert agent._stages_match("conceptual_soundness", "conceptual_soundness")

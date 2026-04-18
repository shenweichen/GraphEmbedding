import importlib.util
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_FILES = [
    "alias.py",
    "deepwalk_wiki.py",
    "line_wiki.py",
    "node2vec_flight.py",
    "node2vec_wiki.py",
    "sdne_wiki.py",
    "struc2vec_flight.py",
]
TF_EXAMPLES = {"line_wiki.py", "sdne_wiki.py"}
GENSIM_EXAMPLES = {"deepwalk_wiki.py", "node2vec_flight.py", "node2vec_wiki.py", "struc2vec_flight.py"}


def load_example_module(example_file):
    module_path = EXAMPLES_DIR / example_file
    spec = importlib.util.spec_from_file_location(f"example_{module_path.stem}", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("example_file", EXAMPLE_FILES)
def test_examples_smoke(example_file):
    if example_file in TF_EXAMPLES:
        pytest.importorskip("tensorflow")
    if example_file in GENSIM_EXAMPLES:
        pytest.importorskip("gensim")
        pytest.importorskip("pandas")
    if example_file == "struc2vec_flight.py":
        pytest.importorskip("fastdtw")

    module = load_example_module(example_file)
    result = module.main(smoke=True, show=False)

    if isinstance(result, dict):
        assert len(result) > 0
    elif isinstance(result, tuple):
        assert all(item is not None for item in result)
    else:
        assert result is not None

import importlib


def test_package_imports():
    pkg = importlib.import_module("animal_data_analysis_tools")
    assert hasattr(pkg, "__version__") and isinstance(pkg.__version__, str)


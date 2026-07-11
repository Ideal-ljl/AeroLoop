# Contributing

Contributions are welcome through issues and pull requests.

1. Keep the core package free of simulator and model-framework imports.
2. Preserve the canonical observation/action protocol and document conversions.
3. Add deterministic tests for behavior changes and extension fixtures.
4. Run `PYTHONPATH=src python -m unittest discover -s tests -v` and
   `ruff check src tests` before submitting.
5. Complete `docs/MODEL_ADAPTER_CHECKLIST.md` for a new model adapter.

By contributing, you agree that your contribution is licensed under Apache-2.0.

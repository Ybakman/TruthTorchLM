from setuptools import setup, find_packages

requirements = ["aiohttp",
                "evaluate",
                "instructor",
                "litellm",
                "nest_asyncio",
                "numpy",
                "outlines",
                "pandas",
                "pydantic",
                "PyYAML",
                "Requests",
                "scikit_learn",
                "scipy",
                "sentence_transformers",
                "termcolor",
                "torch",
                "tqdm",
                "transformers",
                "absl-py",
                "nltk",
                "rouge_score",
                "wandb",
                "sentencepiece",
                "accelerate>=0.26.0",
                # --- TruthTorchLM-HC additions ---
                "datasets",      # dataset loaders (utils/dataset_utils.py)
                "pyarrow",       # Stage-A generation cache (Parquet)
                "matplotlib"]    # accuracy-latency frontier plots


setup(
    # Renamed from upstream "TruthTorchLM" so this research fork can never be confused
    # with, or published over, the upstream distribution. The *import* name is unchanged.
    name="TruthTorchLM-HC",
    version="0.1.19",           # tracks the upstream version this fork is based on
    author="Yavuz Faruk Bakman (upstream); Jason Lucas (HC fork)",
    author_email="ybakman@usc.edu",
    description="TruthTorchLM-HC: a health-coaching research fork of TruthTorchLM, adding a pure black-box UQ benchmark (latency instrumentation, calibration/safety metrics, cached Stage A-D harness, and health datasets).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},         # Maps the base package directory
    # Automatically find and include all packages
    packages=find_packages(where="src"),
    install_requires=requirements,  # List of dependencies
    python_requires=">=3.10",  # Minimum Python version
)

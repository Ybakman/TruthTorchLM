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
                "accelerate>=0.26.0"]


setup(
    name="TruthTorchLM",  # Your package name
    version="0.1.17",           # Package version
    author="Yavuz Faruk Bakman",
    author_email="ybakman@usc.edu",
    description="TruthTorchLM is an open-source library designed to assess truthfulness in language models' outputs. The library integrates state-of-the-art methods, offers comprehensive benchmarking tools across various tasks, and enables seamless integration with popular frameworks like Huggingface and LiteLLM.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},         # Maps the base package directory
    # Automatically find and include all packages
    packages=find_packages(where="src"),
    install_requires=requirements,  # List of dependencies
    python_requires=">=3.10",  # Minimum Python version
)

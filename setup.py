from setuptools import setup, find_packages

requirements = ["aiohttp==3.9.3",
    "evaluate==0.4.1",
    "instructor",
    "litellm==1.57.0",
    "nest_asyncio==1.6.0", 
    "numpy==1.26.4",
    "outlines==0.1.4",
    "pandas==2.2.3",
    "pydantic",
    "PyYAML",
    "Requests==2.32.3",
    "scikit_learn==1.4.1.post1",
    "scipy==1.13.0",
    "sentence_transformers==3.1.1",
    "termcolor==2.5.0",
    "torch==2.2.2", 
    "tqdm==4.66.2",
    "transformers==4.44.2",
    "absl-py",
    "nltk",
    "rouge_score",
    "wandb"]


setup(
    name="TruthTorchLM",  # Your package name
    version="0.1.3",           # Package version
    author="Yavuz Faruk Bakman",
    author_email="ybakman@usc.edu",
    description="TruthTorchLM is an open-source library designed to detect and mitigate hallucinations in text generation models. The library integrates state-of-the-art methods, offers comprehensive benchmarking tools across various tasks, and enables seamless integration with popular frameworks like Huggingface and LiteLLM.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},         # Maps the base package directory
    packages=find_packages(where="src"),  # Automatically find and include all packages
    install_requires=requirements,  # List of dependencies
    python_requires=">=3.10",  # Minimum Python version
)

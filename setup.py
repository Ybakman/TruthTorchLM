from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="TruthTorchLM",  # Your package name
    version="0.1.0",           # Package version
    author="Yavuz Faruk Bakman",
    author_email="ybakman@usc.edu",
    description="A short description of your library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},         # Maps the base package directory
    packages=find_packages(where="src"),  # Automatically find and include all packages
    install_requires=requirements,  # List of dependencies
    python_requires=">=3.10",  # Minimum Python version
)

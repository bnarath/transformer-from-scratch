from setuptools import setup, find_packages

setup(
    name="transformer",
    version="0.1.0",
    packages=find_packages(where="transformer"),
    package_dir={"": "transformer"},
    install_requires=open("requirements.txt").read().splitlines(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)

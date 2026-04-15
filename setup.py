from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements= f.read().splitlines()

setup(
    name="smart-manufacturing-efficiency-prediction",
    version="0.1",
    author="Tannushree Vaishnav",
    packages=find_packages(),
    install_requires=requirements
)
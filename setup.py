from setuptools import setup

with open("README.md", "r") as f:
    long_desciption = f.read()

setup(
    name="graphs-on-grids",
    version="0.0.1",
    description="A High-Level Framework for Graph Neural Networks on power grid data based on TensorFlow2",
    url="https://github.com/th-koeln-iet/graphs-on-grids",
    author="Allen Kletinitch",
    author_email="allen.kletinitch@gmail.com",
    license="MIT",
    project_urls={"documentation": "https://graphs-on-grids.readthedocs.io/"},
    python_requires=">=3.8",
    long_description=long_desciption,
    long_description_content_type="text/markdown",
    packages=[
        "graphs_on_grids",
        "graphs_on_grids.layers",
        "graphs_on_grids.layers.temporal",
        "graphs_on_grids.metrics",
        "graphs_on_grids.preprocessing",
        "graphs_on_grids.structure",
    ],
    install_requires=[
        "numpy~=1.23.5",
        "tensorflow~=2.12.0",
        "pandas~=2.0.1",
        "scikit-learn==1.2.2",
        "pytest==7.3.1",
        "tqdm~=4.65.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

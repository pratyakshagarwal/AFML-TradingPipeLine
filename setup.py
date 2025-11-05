from setuptools import setup, find_packages
from pathlib import Path

# Read README safely
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="AFML-TradingPipeline",
    version="0.1.0",
    author="Pratyaksh Agarwal",
    author_email="pratyakshagarwal93@gmail.com",
    description="Complete Trading Pipeline using AFML as reference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pratyakshagarwal/AFML-TradingPipeline",
    packages=find_packages(exclude=["tests*", "docs*"]),
    install_requires=[
    ],
    python_requires=">=3.8",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)

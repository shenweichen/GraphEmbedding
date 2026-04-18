import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


REQUIRED_PACKAGES = [
    "gensim>=4.0.0",
    "networkx",
    "joblib",
    "fastdtw",
    "tqdm",
    "numpy",
    "scikit-learn",
    "pandas",
    "matplotlib",
]


setuptools.setup(
    name="ge",
    version="0.1.0",
    author="Weichen Shen",
    author_email="weichenswc@163.com",
    url="https://github.com/shenweichen/GraphEmbedding",
    packages=setuptools.find_packages(exclude=[]),
    python_requires=">=3.7",
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "tf": ["tensorflow>=1.15.5"],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "python-coveralls>=2.9.3",
        ],
    },
    entry_points={},
    license="MIT license",
)

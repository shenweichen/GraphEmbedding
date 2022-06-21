import setuptools


with open("README.md", "r") as fh:

    long_description = fh.read()


REQUIRED_PACKAGES = [
    # 'tensorflow>=1.4.0',
    'gensim>=4.0.0',
    'networkx',
    'joblib',
    'fastdtw',
    'tqdm',
    'numpy',
    'scikit-learn',
    'pandas',
    'matplotlib',
    'deepctr'
]


setuptools.setup(

    name="ge",

    version="0.0.0",

    author="Weichen Shen",

    author_email="weichenswc@163.com",

    url="https://github.com/shenweichen/GraphEmbedding",

    packages=setuptools.find_packages(exclude=[]),

    python_requires='>=3.5',  # 3.4.6

    install_requires=REQUIRED_PACKAGES,

    extras_require={

        "cpu": ['tensorflow>=1.4.0,!=1.7.*,!=1.8.*'],

        "gpu": ['tensorflow-gpu>=1.4.0,!=1.7.*,!=1.8.*'],

    },

    entry_points={

    },
    license="MIT license",


)

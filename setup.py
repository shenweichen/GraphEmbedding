import setuptools


with open("README.md", "r") as fh:

    long_description = fh.read()


REQUIRED_PACKAGES = [
    # 'tensorflow>=1.4.0,<=1.12.0',
    'gensim',
    'networkx',
    'scikit-learn',
    'numpy'
]


setuptools.setup(

    name="ge",

    version="0.0.0",

    author="Weichen Shen",

    author_email="wcshen1994@163.com",

    url="https://github.com/shenweichen/GraphEmbedding",

    packages=setuptools.find_packages(exclude=[]),

    python_requires='>=3.4',  # 3.4.6

    install_requires=REQUIRED_PACKAGES,

    extras_require={

        "tf": ['tensorflow>=1.4.0,!=1.7.*,!=1.8.*'],

        "tf_gpu": ['tensorflow-gpu>=1.4.0,!=1.7.*,!=1.8.*'],

    },

    entry_points={

    },
    license="MIT license",


)

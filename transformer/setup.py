from setuptools import setup, find_packages

setup(
    name='my_transformer',
    version='1.0',
    packages=find_packages(),
    author='Haojie Zhang',
    description='Implementation of Transformer from scratch',
    url='',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: Ubuntu22.04',
    ],
    requires=[
        'sentencepiece',
        'argparse'
    ]
)

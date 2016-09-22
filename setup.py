#!/usr/bin/env python
from setuptools import setup
from setuptools import find_packages

setup(
    name='argmining',
    version='0.0.1',
    description='Machine learning environment for paper *What to do with an airport? Mining Arguments in the German Online Participation Project Tempelhofer Feld*',
    url='http://dbs.cs.uni-duesseldorf.de',
    author='HHU Duesseldorf',
    author_email='liebeck@cs.uni-duesseldorf.de',
    packages=find_packages(exclude=['tests']),
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.5',
    ],
    install_requires=[
        'pandas_confusion'
    ]
)

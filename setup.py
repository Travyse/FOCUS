from setuptools import setup, find_packages

setup(
    name='focus',
    version='0.1',
    packages=find_packages(include=['FOCUS', 'FOCUS.*']),
    install_requires=[
        line.strip() for line in open('requirements.txt') if line.strip()
    ],
)
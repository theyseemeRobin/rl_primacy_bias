from setuptools import setup

with open('requirements.txt', 'r') as file:
    requirements = [line.strip() for line in file if line.strip()]

setup(
    name='RLPrimacyBias',
    install_requires=requirements,
)
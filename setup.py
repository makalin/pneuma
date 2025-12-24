# setup.py

from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name="pneuma-bandmate",
    version="0.1.0",
    author="Mehmet T. AKALIN",
    author_email="dev@frange.dev",
    description="PNEUMA: A Synthetic Bandmate",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/frangedev/pneuma",
    packages=find_packages(exclude=["tests"]),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'pneuma=main:main',
        ],
    },
)

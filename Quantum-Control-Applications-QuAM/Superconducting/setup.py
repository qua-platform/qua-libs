from setuptools import setup, find_packages

setup(
    name="quam-components",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "quam @ git+https://github.com/qua-platform/quam.git",
        "qualang-tools>=0.1.0",
    ],
    author="Dean Poulos",
    author_email="dean.poulos@quantum-machines.co",
    description="QuAM Components for Quantum Control",
    license="BSD-3-Clause",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)

from setuptools import setup, find_packages

setup(
    name="aeg",
    version="0.0",
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'tensorflow'
    ]
)

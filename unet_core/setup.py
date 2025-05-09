from setuptools import setup, find_packages

setup(
    name='unet_core',
    version='0.1.0',
    description='UNet implementation for reuse',
    author='Szymon Stasiak',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
        #TODO LATER
    ],
)

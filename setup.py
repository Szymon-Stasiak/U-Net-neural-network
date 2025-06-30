from setuptools import setup, find_packages

setup(
    name='unet_core',
    version='0.1.8',
    packages=find_packages(),
    include_package_data=True,
    url='',
    license='',
    author='Symon Stasiak',
    author_email='',
    description='UNET implementation for reuse with ready glandsModel',
    package_data={
        'unet_core': ['GlandsFinder.pth.tar']
    },
)

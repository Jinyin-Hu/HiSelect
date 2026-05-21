from setuptools import setup, find_packages

setup(
    name='HiSelect',
    version='0.1.0',
    description='High-quality seismic station selector based on SNR, CC, and azimuthal coverage',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'obspy',
        'scikit-learn',
        'matplotlib',
    ],
    extras_require={
        'pysep':    ['pysep'],
        'cartopy':  ['cartopy'],
    },
)

from setuptools import find_packages, setup

setup(
    name='DLNest',
    version='0.2.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'nvidia-ml-py>=375.53.1',
        'APScheduler>=3.6.3',
        'torch>=1.3.0',
        'prompt-toolkit>=3.0.7'
    ],
    package_data={
        "DLNest":["FactoryFiles/*"]
    }
)
from setuptools import find_packages, setup

setup(
    name='DLNest',
    version='0.3.2',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'nvidia-ml-py3',
        'APScheduler>=3.6.3',
        'torch',
        'prompt-toolkit>=3.0.7',
        'tornado>=6.0'
    ],
    package_data={
        "DLNest":[
            "FactoryFiles/*",
            "FactoryFiles/*/*",
            "FactoryFiles/*/*/*"
        ]
    },
    python_requires=">=3.6" 
)

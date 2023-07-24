from setuptools import setup

setup(
    name='msintuit',
    version='0.0.1',
    install_requires=[
        "torch",
        "torchvision",
    ],
    packages=["msintuit"],
    package_dir={"msintuit": "msintuit"}
)

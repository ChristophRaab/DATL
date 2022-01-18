import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

required_packages = [
    "torch", "torchvision", "scikit-learn", "pandas", "numpy", "matplotlib",
    "kmeans_pytorch", "pytorch_metric_learning", "opencv-python", "Pillow",
    "torchsummary"
]
setuptools.setup(
    name="datl",
    version="0.1",
    author="Christoph Raab",
    author_email="christophraab@outlook.de",
    description=
    "DATL: Source Code for the paper Domain Adversarial Tangent Subspace Alignment for explainable domain adaptation.",
    license="MIT",
    url="https://github.com/ChristophRaab/datl",
    packages=setuptools.find_packages(include=['datl', 'datl.*']),
    python_requires=">=3.8",
    install_requires=required_packages,
    package_data={"": ["README.md", "LICENSE"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ])

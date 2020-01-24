import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="forcsensei",
    version="1.1.0",
    author="FORCaist",
    author_email="FORCaist.user@gmail.com",
    description="The FORCsensei package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FORCaist",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="forcsensei",
    version="0.0.2",
    author="FORCaist",
    author_email="FORCaist.user@gmail.com",
    description="A small example of forcsensei",
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
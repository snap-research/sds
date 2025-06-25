from setuptools import setup, find_packages

def get_requirements(filename="requirements.txt"):
    """
    Reads the requirements from a requirements.txt file and returns a list of dependencies.
    """
    requirements = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            # Ignore comments, blank lines, and editable installs
            if line and not line.startswith("#") and not line.startswith("-e"):
                requirements.append(line)
    return requirements

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A small example package"


setup(
    name="streaming-dataset",
    version="0.0.1",
    author="Ivan Skorokhodov",
    author_email="iskorokhodov@snap.com",
    description="Streaming Dataset library for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.sc-corp.net/Snapchat/sds",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(), # Required
    python_requires=">=3.11",
    install_requires=get_requirements(),
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
)

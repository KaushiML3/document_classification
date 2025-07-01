import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Document-Classification-Project"
AUTHOR_USER_NAME = "kaushi"
SRC_REPO = "documentClassifire"
AUTHOR_EMAIL = "Kaushigihan@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/",
    package_dir={"": "documentClassifire"},
    packages=setuptools.find_packages(where="documentClassifire")
)
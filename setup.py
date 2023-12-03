import os
import setuptools

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_version():
    with open(os.path.join(_CURRENT_DIR, "hj_reachability", "__init__.py")) as f:
        for line in f:
            if line.startswith("__version__") and "=" in line:
                version = line[line.find("=") + 1:].strip(" '\"\n")
                if version:
                    return version
        raise ValueError("`__version__` not defined in `hj_reachability/__init__.py`")


def _parse_requirements(file):
    with open(os.path.join(_CURRENT_DIR, file)) as f:
        return [line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))]


setuptools.setup(name="hj_reachability",
                 version=_get_version(),
                 description="Hamilton-Jacobi reachability analysis in JAX.",
                 long_description=open("README.md").read(),
                 long_description_content_type="text/markdown",
                 author="Ed Schmerling",
                 author_email="ednerd@gmail.com",
                 url="https://github.com/StanfordASL/hj_reachability",
                 license="MIT",
                 packages=setuptools.find_packages(),
                 install_requires=_parse_requirements("requirements.txt"),
                 tests_require=_parse_requirements("requirements-test.txt"),
                 python_requires="~=3.8")

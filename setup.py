from setuptools import setup, find_packages

setup(
    name="nirvana",
    version="0.1.0",
    description="Making an AI that is that the level of a Junior AI engineer",
    author="Fangyuan Yu",
    author_email="fangyuan.yu@temus.com",
    url="https://github.com/tplusplusdevhub/Nirvana_auto_alignment_project",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "notebooks": ["jupyter", "matplotlib", "seaborn"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)

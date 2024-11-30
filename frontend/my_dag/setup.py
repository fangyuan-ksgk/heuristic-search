from setuptools import setup, find_packages

setup(
    name="tiny_dag",
    version="0.2",
    package_dir={"": "package"},
    packages=find_packages(where="package"),
    install_requires=[
        "fastapi",
        "uvicorn",
        "websockets",
        "honcho",
    ],
    entry_points={
        "console_scripts": [
            "tiny_dag_backend=tiny_dag.api.main:run_server",
            "tiny_dag_frontend=tiny_dag.cli.dev:run_frontend",
            "serve_tiny_dag=tiny_dag.cli.dev:run_dev",
        ],
    },
)
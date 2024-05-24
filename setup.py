"""
Setup.py
"""

from setuptools import find_packages, setup
import platform

# Detect the operating system
is_mac = platform.system() == "Darwin"

# Select the appropriate requirements file
requirements_file = "requirements-mac.txt" if is_mac else "requirements.txt"

# Load the requirements from the selected file
requirements = []
with open(requirements_file, "r", encoding="utf-8") as req_file:
    for item in req_file:
        requirements.append(item.strip())

setup(
    name="yolollm",
    description="YOLOLLM ",
    version="1.0.0",
    author="Daniel Montano",
    author_email="hello@danielmontano.io",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    setup_requires=["wheel"],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "test-with-video=src.bin.test_video:run",
            "main-loop=src.bin.main_loop:run",
        ],
    },
)

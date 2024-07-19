# -*- coding:utf-8 -*-
# @FileName  :setup.py
# @Time      :2024/7/19 09:23
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com


import os
from pathlib import Path

from setuptools import find_namespace_packages, setup

dirname = Path(os.path.dirname(__file__))
version = "v1.0.0"

requirements = {
    "install": [
        "setuptools<=65.0",
        "numpy==1.24.4",
        "kaldi_native_fbank",
        "onnxruntime==1.18.0",
        "sentencepiece==0.2.0",
        "soundfile==0.12.1",
        "huggingface_hub",
    ]
}

install_requires = requirements["install"]


setup(
    name="sensevoice-onnx",
    version=version,
    url="https://github.com/lovemefan/SenseVoice-python",
    author="Lovemefan",
    author_email="lovemefan@outlook.com",
    description="SenseVoice-python: A enterprise-grade open source multi-language asr system from funasr opensource "
    "with onnxruntime",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="The MIT License",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.8.0",
    entry_points={"console_scripts": ["sensevoice=sensevoice.sense_voice:main"]},
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

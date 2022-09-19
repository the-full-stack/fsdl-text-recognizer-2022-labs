"""Fixes issue where Poetry ignores environment markers when using local path.

See https://github.com/python-poetry/poetry/issues/2765

Usage:
    $ poetry lock
    $ python add_markers.py
"""
from __future__ import annotations

with open("poetry.lock", encoding="utf-8") as f:
    contents = f.read()

DARWIN = """marker = "sys_platform == 'darwin'"
"""

LINUX = """marker = "sys_platform == 'linux'"
"""

S1 = """
[[package]]
name = "torch"
version = "1.12.1"
description = "Tensors and Dynamic neural networks in Python with strong GPU acceleration"
category = "main"
optional = false
python-versions = ">=3.7.0"
"""

S2 = """
[[package]]
name = "torch"
version = "1.12.1+cu116"
description = "Tensors and Dynamic neural networks in Python with strong GPU acceleration"
category = "main"
optional = false
python-versions = ">=3.7.0"
"""

S3 = """
[[package]]
name = "torchvision"
version = "0.13.1"
description = "image and video datasets and models for torch deep learning"
category = "main"
optional = false
python-versions = ">=3.7"
"""

S4 = """
[[package]]
name = "torchvision"
version = "0.13.1+cu116"
description = "image and video datasets and models for torch deep learning"
category = "main"
optional = false
python-versions = ">=3.7"
"""

for [substring, insert] in [
    (S1, DARWIN),
    (S2, LINUX),
    (S3, DARWIN),
    (S4, LINUX),
]:
    position = contents.find(substring) + len(substring)
    contents = contents[:position] + insert + contents[position:]


with open("poetry.lock", "w", encoding="utf-8") as f:
    f.write(contents)

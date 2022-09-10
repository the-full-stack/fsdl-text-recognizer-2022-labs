"""Tests for the text_recognizer.callbacks.util module."""
import random
import string
import tempfile

import pytorch_lightning as pl

from text_recognizer.callbacks.util import check_and_warn


def test_check_and_warn_simple():
    """Test the success and failure in the case of a simple class we control."""

    class Foo:
        pass  # a class with no special attributes

    letters = string.ascii_lowercase
    random_attribute = "".join(random.choices(letters, k=10))
    assert check_and_warn(Foo(), random_attribute, "random feature")
    assert not check_and_warn(Foo(), "__doc__", "feature of all Python objects")


def test_check_and_warn_tblogger():
    """Test that we return a truthy value when trying to log tables with TensorBoard.

    We added check_and_warn in order to prevent a crash if this happens.
    """
    tblogger = pl.loggers.TensorBoardLogger(save_dir=tempfile.TemporaryDirectory())
    assert check_and_warn(tblogger, "log_table", "tables")


def test_check_and_warn_wandblogger():
    """Test that we return a falsy value when we try to log tables with W&B.

    In adding check_and_warn, we don't want to block the feature in the happy path.
    """
    wandblogger = pl.loggers.WandbLogger(anonymous=True)
    assert not check_and_warn(wandblogger, "log_table", "tables")

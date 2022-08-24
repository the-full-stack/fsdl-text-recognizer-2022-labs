"""Utilities for model development scripts: training and staging."""
import argparse
import importlib

DATA_CLASS_MODULE = "text_recognizer.data"
MODEL_CLASS_MODULE = "text_recognizer.models"


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def setup_data_and_model_from_args(args: argparse.Namespace):
    data_class = import_class(f"{DATA_CLASS_MODULE}.{args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{args.model_class}")

    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    return data, model

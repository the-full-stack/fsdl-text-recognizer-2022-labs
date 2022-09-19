"""Test for data.iam module."""
from text_recognizer.data.iam import IAM


def test_iam_parsed_lines():
    """Tests that we retrieve the same number of line labels and line image cropregions."""
    iam = IAM()
    iam.prepare_data()
    for iam_id in iam.all_ids:
        assert len(iam.line_strings_by_id[iam_id]) == len(iam.line_regions_by_id[iam_id])


def test_iam_data_splits():
    """Fails when any identifiers are shared between training, test, or validation."""
    iam = IAM()
    iam.prepare_data()
    assert not set(iam.train_ids) & set(iam.validation_ids)
    assert not set(iam.train_ids) & set(iam.test_ids)
    assert not set(iam.validation_ids) & set(iam.test_ids)

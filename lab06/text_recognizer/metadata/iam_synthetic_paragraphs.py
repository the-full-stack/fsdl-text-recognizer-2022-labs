import text_recognizer.metadata.iam_paragraphs as iam_paragraphs
import text_recognizer.metadata.shared as shared

NEW_LINE_TOKEN = iam_paragraphs.NEW_LINE_TOKEN

PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "iam_synthetic_paragraphs"

EXPECTED_BATCH_SIZE = 64
EXPECTED_GPUS = 8
EXPECTED_STEPS = 40

# set the dataset's length based on parameters during typical training
DATASET_LEN = EXPECTED_BATCH_SIZE * EXPECTED_GPUS * EXPECTED_STEPS

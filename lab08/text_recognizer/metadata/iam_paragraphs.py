import text_recognizer.metadata.emnist as emnist
import text_recognizer.metadata.shared as shared


PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "iam_paragraphs"

NEW_LINE_TOKEN = "\n"
MAPPING = [*emnist.MAPPING, NEW_LINE_TOKEN]

# must match IMAGE_SCALE_FACTOR for IAMLines to be compatible with synthetic paragraphs
IMAGE_SCALE_FACTOR = 2
IMAGE_HEIGHT, IMAGE_WIDTH = 576, 640
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

MAX_LABEL_LENGTH = 682

DIMS = (1, IMAGE_HEIGHT, IMAGE_WIDTH)
OUTPUT_DIMS = (MAX_LABEL_LENGTH, 1)

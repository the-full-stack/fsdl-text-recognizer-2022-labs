from pathlib import Path

import text_recognizer.metadata.emnist as emnist
import text_recognizer.metadata.shared as shared

PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "emnist_lines"
ESSENTIALS_FILENAME = Path(__file__).parents[1].resolve() / "data" / "emnist_lines_essentials.json"


CHAR_HEIGHT, CHAR_WIDTH = emnist.DIMS[1:3]
DIMS = (emnist.DIMS[0], CHAR_HEIGHT, None)  # width variable, depends on maximum sequence length

MAPPING = emnist.MAPPING

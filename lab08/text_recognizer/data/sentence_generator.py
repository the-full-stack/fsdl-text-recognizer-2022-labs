"""SentenceGenerator class and supporting functions."""
import itertools
import re
import string
from typing import List, Optional

import nltk
import numpy as np

from text_recognizer.data.base_data_module import BaseDataModule

NLTK_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "nltk"


class SentenceGenerator:
    """Generate text sentences using the Brown corpus."""

    def __init__(self, max_length: Optional[int] = None):
        self.text = brown_text()
        self.word_start_inds = [0] + [_.start(0) + 1 for _ in re.finditer(" ", self.text)]
        self.max_length = max_length

    def generate(self, max_length: Optional[int] = None) -> str:
        """Sample a string from text of the Brown corpus of length at least one word and at most max_length."""
        if max_length is None:
            max_length = self.max_length
        if max_length is None:
            raise ValueError("Must provide max_length to this method or when making this object.")

        sampled_text, num_tries = None, 0
        while (not sampled_text) and (num_tries <= 10):  # try several times to generate sample text
            first_ind = np.random.randint(0, len(self.word_start_inds) - 1)
            start_ind = self.word_start_inds[first_ind]

            end_ind_candidates = self._get_end_ind_candidates(first_ind, start_ind, max_length)

            if len(end_ind_candidates) == 0:  # sampling failed, try again
                num_tries += 1
                continue
            else:
                end_ind = np.random.choice(end_ind_candidates)
                sampled_text = self.text[start_ind:end_ind].strip()

        if sampled_text is not None:
            return sampled_text
        else:
            raise RuntimeError("Was not able to generate a valid string")

    def _get_end_ind_candidates(self, first_ind: int, start_ind: int, max_length: int) -> List[int]:
        end_ind_candidates = []
        for ind in range(first_ind + 1, len(self.word_start_inds)):
            if self.word_start_inds[ind] - start_ind > max_length:
                break
            end_ind_candidates.append(self.word_start_inds[ind])
        return end_ind_candidates


def brown_text():
    """Return a single string with the Brown corpus with all punctuation stripped."""
    sents = load_nltk_brown_corpus()
    text = " ".join(itertools.chain.from_iterable(sents))
    text = text.translate({ord(c): None for c in string.punctuation})
    text = re.sub("  +", " ", text)
    return text


def load_nltk_brown_corpus():
    """Load the Brown corpus using the NLTK library."""
    nltk.data.path.append(NLTK_DATA_DIRNAME)
    try:
        nltk.corpus.brown.sents()
    except LookupError:
        NLTK_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        nltk.download("brown", download_dir=NLTK_DATA_DIRNAME)
    return nltk.corpus.brown.sents()

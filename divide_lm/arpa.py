from __future__ import annotations

import logging
import os
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, TextIO, Tuple

from tqdm import tqdm

DELIMITER = "\t"
_UNK_lgp = -100
_UNK_lgbo = 0.0

logging.basicConfig(level=logging.INFO)


@dataclass
class Weight:
    lgp: float
    lgbo: float = 0.0

    def __call__(self):
        return self.lgp, self.lgbo


class ARPAModel:
    def __init__(
        self,
        filename: str = None,
        unk_weight=Weight(_UNK_lgp, _UNK_lgbo),
        numerator: ARPAModel = None,
    ):
        super().__init__()
        self.weight_dict = OrderedDict()
        self.numerator = numerator
        self.new_keys = []
        self.counts = []
        self._unk_weight = unk_weight
        self.bos = "<s>"
        self.eos = "</s>"
        if filename is not None:
            assert os.path.isfile(filename)
            self.filename = filename
            self.load_arpa()

    def load_arpa(self) -> None:
        assert os.path.isfile(self.filename)
        with open(self.filename) as f:
            counts = self.read_ngram_counts(f)
            if self.numerator is not None:
                for _ in range(len(counts)):
                    self.new_keys.append([])

            self.counts = counts
            self.model_order = len(self.counts)

            for order in range(1, self.model_order + 1):
                self.read_ngram(f, order)

            self.read_end(f)

    def read_ngram_counts(self, f: TextIO) -> List[int]:
        counts = []
        line = self.read_with_skipping_empty_lines(f)
        assert line == "\\data\\\n"
        for line in f:
            if line.strip() == "":
                break
            assert line.startswith("ngram"), line
            counts.append(int(line.strip()[8:]))
        return counts

    def reading_prompt(self, order: int) -> None:
        logging.info(f"Loading {order}-gram entries from {self.filename}")

    @staticmethod
    def read_with_skipping_empty_lines(f: TextIO) -> str:
        line = f.readline()
        while line.strip() == "":
            line = f.readline()
        return line

    def read_ngram(
        self,
        f: TextIO,
        order: int,
    ) -> None:
        self.reading_prompt(order)
        self.read_ngram_header(f, order)

        for _ in tqdm(range(self.counts[order - 1])):
            line = f.readline()
            lgp_key_lgbo = line.strip().split(DELIMITER)
            key = lgp_key_lgbo[1]
            lgp = float(lgp_key_lgbo[0])
            lgbo = 0.0 if len(lgp_key_lgbo) == 2 else float(lgp_key_lgbo[2])
            self.weight_dict[key] = Weight(lgp, lgbo)
            if (
                self.numerator is not None
                and key not in self.numerator.weight_dict
            ):
                self.new_keys[order - 1].append(key)

    def read_ngram_header(self, f: TextIO, cur_order: int) -> None:
        line = self.read_with_skipping_empty_lines(f)
        assert (
            line == f"\\{cur_order}-grams:\n"
        ), f"Current line is: {line} while \\{cur_order}-grams: is expected"

    @staticmethod
    def read_end(f: TextIO) -> None:
        line = f.readline()
        while line.strip() == "":
            line = f.readline()
        assert (
            line == "\\end\\\n"
        ), f"Current line is: {line} while \\end\\ is expected"

    def weight(self, ngram: str) -> Weight:
        if ngram in self.weight_dict:
            return self.weight_dict[ngram]
        else:
            self._unk_weight

    def score(self, ngram: str) -> float:
        if len(ngram.split()) > self.model_order:
            ngram = " ".join(ngram.split()[-self.model_order :])

        if ngram in self.weight_dict:
            lgp, lgbo = self.weight_dict[ngram]()
            return lgp
        elif len(ngram.split()) == 1:
            lgp, lgbo = self._unk_weight()
            return lgp
        else:
            ngram_list = ngram.split()
            ngram_prefix = " ".join(ngram_list[:-1])
            ngram_suffix = " ".join(ngram_list[1:])
            lgbo = (
                0.0
                if ngram_prefix not in self.weight_dict
                else self.weight_dict[ngram_prefix]()[1]
            )
            lgp = self.score(ngram_suffix)
            return lgbo + lgp

    def full_scores(self, sentence: str, bos=True, eos=True) -> List[float]:
        scores = []
        words_list = []
        if bos:
            words_list.append(self.bos)

        words_list.extend(sentence.split())
        if eos:
            words_list.append(self.eos)
        num_words = len(words_list)
        for i in range(1, num_words + 1):
            scores.append(self.score(" ".join(words_list[:i])))
        if bos:
            # Make it compitable to kenlm.full_scores
            scores = scores[1:]
        return scores

    def iterate_all_entries(
        self,
    ) -> Tuple[float, str, float]:
        for key, weight in self.weight_dict.items():
            lgp = weight.lgp
            lgbo = weight.lgbo
            yield lgp, key, lgbo

    def iterate_ngram(
        self,
        order: int,
        all_entries_iterable: Iterable[Tuple[float, str, float]],
    ) -> Tuple[float, str, float]:
        count = self.counts[order - 1]
        for _, (lgp, key, lgbo) in zip(range(count), all_entries_iterable):
            yield lgp, key, lgbo

    @staticmethod
    def compile_ngram_entry(lgp: float, key: str, lgbo: float) -> str:
        if lgbo != 0.0:
            ngram_entry = f"{lgp}{DELIMITER}{key}{DELIMITER}{lgbo}"
        else:
            ngram_entry = f"{lgp}{DELIMITER}{key}"

        return ngram_entry

    def save_ngram(
        self,
        order: int,
        all_entries_iterable: Iterable[Tuple[float, str, float]],
        out_f: TextIO,
    ) -> None:
        print(f"\n\\{order}-grams:", file=out_f)
        for lgp, key, lgbo in self.iterate_ngram(order, all_entries_iterable):
            ngram_entry = self.compile_ngram_entry(lgp, key, lgbo)
            print(f"{ngram_entry}", file=out_f)

    def write_header(self, out_f: TextIO) -> None:
        print("\\data\\", file=out_f)
        for i, count in enumerate(self.counts):
            print(f"ngram {i+1}={count}", file=out_f)

    def save(self, out_path: str) -> None:

        with open(out_path, "w") as out_f:
            self.write_header(out_f)
            all_entries_iterable = self.iterate_all_entries()
            for order in range(1, self.model_order + 1):
                self.save_ngram(order, all_entries_iterable, out_f)
            print("\n\\end\\", file=out_f)

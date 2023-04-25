import copy
import logging
import os
from typing import Optional

from tqdm import tqdm

from .arpa import ARPAModel, Weight


class LMConfig:
    def __init__(self, model_path: str, unk_weight: Weight):
        assert os.path.isfile(model_path)
        self.model_path = model_path
        self.unk_weight = unk_weight


class Divider:
    def __init__(
        self,
        numerator_config: str,
        denominator_config: str,
    ):
        super().__init__()
        self.numerator_config = numerator_config
        self.denominator_config = denominator_config

    def load_models(
        self,
    ) -> None:
        self.numerator = ARPAModel(
            filename=self.numerator_config.model_path,
            unk_weight=self.numerator_config.unk_weight,
        )
        self.denominator = ARPAModel(
            filename=self.denominator_config.model_path,
            unk_weight=self.denominator_config.unk_weight,
            numerator=self.numerator,
        )
        # TODO: Warning
        # if self.numerator.model_order < self.denominator.model_order

    def divide(
        self,
        wnum: float,
        wden: float,
        saved_path: str = None,
        return_divided_model: bool = False,
    ) -> Optional[ARPAModel]:
        assert saved_path is not None or return_divided_model
        ret = ARPAModel()
        ret.model_order = self.numerator.model_order
        ret.counts = copy.deepcopy(self.numerator.counts)
        for order in range(1, self.denominator.model_order + 1):
            number_added_grams = len(self.denominator.new_keys[order - 1])
            if number_added_grams > 0:
                ret.counts[order - 1] += number_added_grams

        out_f = None
        if saved_path is not None:
            out_f = open(saved_path, "w")
            ret.write_header(out_f)

        def save_ngram_entry():
            if out_f:
                ngram_entry = ret.compile_ngram_entry(new_lgp, key, new_lgbo)
                print(f"{ngram_entry}", file=out_f)
            if return_divided_model:
                ret.weight_dict[key] = Weight(new_lgp, new_lgbo)

        num_all_entries_iterable = self.numerator.iterate_all_entries()
        for order in range(1, self.numerator.model_order + 1):
            logging.info(f"Processing {order}-gram entries.")
            if out_f:
                print(f"\n\{order}-grams:", file=out_f)
            count = self.numerator.counts[order - 1]
            for num_lgp, key, num_lgbo in tqdm(
                self.numerator.iterate_ngram(order, num_all_entries_iterable),
                total=count,
            ):
                den_lgp = self.denominator.score(key)
                new_lgp = wnum * num_lgp - wden * den_lgp

                new_lgbo = wnum * num_lgbo
                if order < self.numerator.model_order:
                    if key in self.denominator.weight_dict:
                        den_lgbo = self.denominator.weight_dict[key].lgbo
                        new_lgbo = new_lgbo - wden * den_lgbo
                save_ngram_entry()

            if (
                order <= self.denominator.model_order
                and len(self.denominator.new_keys[order - 1]) > 0
            ):
                logging.info(f"Add new {order}-gram entries.")
                number_added_grams = len(self.denominator.new_keys[order - 1])
                for key in tqdm(
                    self.denominator.new_keys[order - 1],
                    total=number_added_grams,
                ):
                    num_lgp = self.numerator.score(key)
                    den_lgp, den_lgbo = self.denominator.weight_dict[key]()
                    new_lgp = wnum * num_lgp - wden * den_lgp
                    new_lgbo = -wden * den_lgbo
                    save_ngram_entry()

        if out_f:
            print("\n\end\\", file=out_f)
            out_f.close()
        return ret


def divide(
    numerator_config: LMConfig,
    denominator_config: LMConfig,
    weight_numerator: float,
    weight_denominator: float,
    saved_path: str,
) -> None:
    assert weight_numerator > 0
    assert weight_denominator > 0
    divider = Divider(numerator_config, denominator_config)
    divider.load_models()
    divider.divide(weight_numerator, weight_denominator, saved_path)

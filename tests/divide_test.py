import kenlm
import numpy as np

from divide_lm import ARPAModel, LMConfig, Weight, divide

l4f = "tests/arpa_files/4gram.arpa"
l2f = "tests/arpa_files/2gram.arpa"
wnum = 0.4
wden = 0.2
ret_f = f"tests/wnum_{wnum}-wden_{wden}-divided.arpa"

unk_lgp = -100.0
unk_lgbo = 0.0
num_config = LMConfig(model_path=l4f, unk_weight=Weight(unk_lgp, unk_lgbo))
den_config = LMConfig(model_path=l2f, unk_weight=Weight(unk_lgp, unk_lgbo))

# No return value.
# The divided language model will be saved to disk directly.
divide(
    numerator_config=num_config,
    denominator_config=den_config,
    weight_numerator=wnum,
    weight_denominator=wden,
    saved_path=ret_f,
)

sentence = "a d b c d c b a"

lm4 = kenlm.Model(l4f)
lm2 = kenlm.Model(l2f)
divided_lm4 = ARPAModel(ret_f)

s4 = np.array(
    [entry[0] for entry in lm4.full_scores(sentence, bos=True, eos=True)]
)
s2 = np.array(
    [entry[0] for entry in lm2.full_scores(sentence, bos=True, eos=True)]
)
ref = wnum * s4 - wden * s2

hyp = divided_lm4.full_scores(sentence)
assert np.allclose(ref, hyp), f"{ref} {hyp}"
print("Done.")

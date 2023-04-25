# Divide a higher order language model with a lower order language model

## Usage

#### Installation
```shell script
git clone git@github.com:k2-fsa/divide_lm.git
python3 setup.py install

```

#### Fake example for demonstating functionality only.
```python
# Full example is in tests/divide_test.py
# Following statements are not real code.
# Express in this way to emphasize the functionality.

l4f = "tests/arpa_files/4gram.arpa"
l2f = "tests/arpa_files/2gram.arpa"
w4 = 0.4
w2 = 0.2

# Not real code.
divided_lm4 = processes_to_divide_lms(l4f, l2f, w4, w2)

lm4 = kenlm.Model(l4f)
lm2 = kenlm.Model(l2f)

sentence = "a d b c d c b a"
# Not real code.
score4_for_each_token = lm4.full_scores(sentence)
score2_for_each_token = lm2.full_scores(sentence)

ref = w4 * score4_for_each_token - w2 * score2_for_each_token
hyp = divided_lm4.full_scores(sentence)
assert np.allclose(ref, hyp), f"{ref} {hyp}"
```

#### Real code of previous example.
```python
# Copied from tests/divide_test.py
# kenlm and numpy is only needed by this test.
# Package divide_lm itself does not depends them.
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

# Divide a higher order language model with a lower order language model

## Usage

#### Installation
```shell script
git clone git@github.com:k2-fsa/divide_lm.git

```

#### Example
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
s4 = lm4.full_scores(sentence)
s2 = lm2.full_scores(sentence)

ref = s4 * w4 - s2 * w2
hyp = divided_lm4.full_scores(sentence)
assert np.allclose(ref, hyp), f"{ref} {hyp}"
```


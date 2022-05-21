## Don't actually simulate the NFST

Instead, we modify the NFST `T` so that the _preimage_ only contains the one input `i` we want. This is done by composing `Id(i) . T`. Then the _image_ of the NFST will actually be the regular language containing everything we want.

This image can be extracted into an NFA, transformed into a DFA, then iterated upon lazily.

Similarly, we can add constraints to the NFST by composing `Id(R)` where `R` is any regular language on either side.

## Problems:

- How to track alphabet? That's important for computing things like complements correctly.
- One easy solution is to require passing in the alphabet in arguments that require it (e.g. complement)

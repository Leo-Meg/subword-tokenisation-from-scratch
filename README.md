# Subword tokenisation, written by hand

Byte Pair Encoding and WordPiece implemented from scratch, with no dependencies, then put to the test.

**Léo Mégret** — implementation, French technical notes.  
**Haeeul Hwang** — English documentation, Korean documentation, review.

> **Repository status** — step 2 of 4, landed by Léo Mégret.

---

## Why this repository

A language model does not receive text, it receives a sequence of integers.
Whatever the tokeniser throws away, the model cannot recover afterwards.

This is the part of the processing chain Léo learned the most from in his first
Master's year, and it is taken up here by writing it by hand. A tokeniser called
from a library is not a tokeniser you understand.

The question behind the algorithm is a linguist's one. Does this system treat all
languages equally.

---

## Steps

| | Step | Modules | Tests | Landed by | Detailed note (French) |
|---|---|---|---:|---|---|
| **1** | Byte Pair Encoding written by hand | `src/bpe.py` | 15 | Léo | [`byte-pair-encoding`](notes/byte-pair-encoding.md) |
| **2** | WordPiece, and why the merge criterion changes everything | `src/wordpiece.py` | 12 | Léo | [`wordpiece`](notes/wordpiece.md) |

**27** tests in total.

---

## What each step found

**1. Byte Pair Encoding written by hand**  
With no dictionary and no annotation, BPE recovers morphemes. The first merges learned on a French corpus are the plural `s`, then `e`, `es`, `en`, `ent`.

**2. WordPiece, and why the merge criterion changes everything**  
Same corpus, same vocabulary size, one line of difference in the criterion: the Jaccard overlap between the two vocabularies is 0.18.

---

## Running it

```bash
python -m src.wordpiece

python -m tests.test_bpe
python -m tests.test_wordpiece
```

---

## How we work

**Nothing to download.** The corpus is written into the code. All of
Léo's Master's notebooks began with a `wget` to a university server or a Google
Drive mount. Two years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in the original coursework are quoted, not erased.** Where a result
that was handed in was wrong or incomplete, we say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what we
expected, we write down what we found.

**The code is commented in French**, because that is the language it was thought
out in. The documentation is in English, and in Korean where Haeeul has caught
up.

---

*Français : [README_FR.md](README_FR.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)*

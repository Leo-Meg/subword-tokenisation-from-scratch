# Subword tokenisation, written by hand

Byte Pair Encoding implemented from scratch, with no dependencies, then put to the test.

**Léo Mégret**, MSc Computational Linguistics, Université Paris Cité

> **Repository status, version 1.** This is the first step of work I am doing in
> stages, each in its own folder. Only version 1 exists so far. I publish as I go
> rather than once everything is finished.

---

## Why this repository

A language model does not receive text, it receives a sequence of integers.
Whatever the tokeniser throws away, the model cannot recover afterwards.

This is the part of the processing chain I learned the most from in my first
Master's year, and I take it up here by writing it by hand. A tokeniser called
from a library is not a tokeniser you understand.

The question I am really after, behind the algorithm, is a linguist's one. Does
this system treat all languages equally.

---

## What exists today

### Version 1, Byte Pair Encoding written by hand

The founding algorithm of modern tokenisation, the one used by GPT, RoBERTa and
LLaMA.

| File | What I do in it |
|---|---|
| `src/bpe.py` | A complete `TokeniseurBPE`. Pre-segmentation, pair counting, merge learning, segmenting a word by merge order, invertible encoding and decoding, vocabulary inspection. |
| `tests/test_bpe.py` | 15 tests, including invertibility and determinism. |

Academic origin. A first-year Master's pair project, implementing BPE and
WordPiece from scratch, training them on French corpora, comparing their
vocabularies and runtimes.

---

## Running the code

```bash
cd 1.tokenisation_python_projet
python -m src.bpe
python -m tests.test_bpe
```

No dependencies, not even NumPy.

---

## What I take from this step

**BPE does not do morphology.** It splits `manger` into `mang` and `er`, which is
right, and `mer` into `m` and `er`, which is not. The algorithm draws no
distinction between the two cases, it merges frequent pairs.

**An unknown word stays segmentable.** That is what makes subwords useful compared
with a whole-word vocabulary, where any unseen word becomes `[UNK]`. It is also
why GPT-2 works on bytes. With 256 possible bytes, no text is out of vocabulary,
in any language and for any emoji.

**My implementation is naive and I own that.** It recounts every pair at each
merge, which is O(V × M). Serious implementations maintain the counts
incrementally. I am keeping the readable version.

---

## What is still open

Nothing in the algorithm distinguishes a real morphological boundary from a
statistical coincidence. I do not know whether another family of tokenisers does
better on this point, nor how one would measure the difference.

The question of fairness between languages is wide open. A tokeniser trained
mostly on English cuts Finnish or Turkish into very short pieces. I would like to
put a number on that, and I am not there yet.

---

## How I work

Four rules I set myself at the start, and intend to keep across the whole
repository.

**Nothing to download.** The corpus is written into the code. All of my Master's
notebooks began with a `wget` to a university server or a Google Drive mount. Two
years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in my coursework are quoted, not erased.** Where a result I handed in
was wrong or incomplete, I say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what I
expected, I write down what I found.

**The code is commented in French.**

---

*French version, which I wrote first, [README_FR.md](README_FR.md).*

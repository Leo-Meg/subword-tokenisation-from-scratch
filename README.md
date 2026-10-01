# Subword tokenisation, written by hand

Byte Pair Encoding implemented from scratch, with no dependencies, then put to the test.

**Léo Mégret** — MSc Computational Linguistics, Université Paris Cité

> **Repository status: version 1.** This is the first step of a piece of work I
> am doing in stages, each in its own folder. Only version 1 exists so far. I am
> publishing as I go rather than once everything is finished, because the point
> of this work is precisely the way one question leads to the next.

---

## Why this repository

A language model never sees text. It sees a sequence of integers. Whatever the
tokeniser throws away, the model can never recover.

It is the most underrated object in the whole processing chain, and the one I
learned the most from in my first Master's year. I take it up again here by
writing it by hand, because a tokeniser called from a library is not a tokeniser
you understand.

The question I am really after, behind the algorithm, is a linguist's one: does
this system treat all languages equally?

---

## What exists today

### Version 1 — Byte Pair Encoding, written by hand

The founding algorithm of modern tokenisation, the one used by GPT, RoBERTa and
LLaMA.

| File | What I do in it |
|---|---|
| `src/bpe.py` | A complete `TokeniseurBPE`: pre-segmentation, pair counting, merge learning, segmenting a word by merge order, invertible encoding and decoding, vocabulary inspection. |
| `tests/test_bpe.py` | 15 tests, including invertibility and determinism. |

Academic origin: a first-year Master's pair project, implementing BPE and
WordPiece from scratch, training them on French corpora and comparing their
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

**BPE is not morphology.** It will happily split `manger` into `mang` and `er`,
which is right, and `mer` into `m` and `er`, which is not right at all. The
algorithm draws no distinction between the two: it merges frequent pairs, and
nothing else.

**An unknown word stays segmentable.** That is the property that makes subwords
worthwhile compared with a whole-word vocabulary, where any unseen word becomes
`[UNK]`, a permanent hole. It is also why GPT-2 works on bytes: with 256
possible bytes, no text is ever out of vocabulary, in any language and for any
emoji.

**My implementation is naive, and I own that.** It recounts every pair at each
merge, which is O(V × M). Serious implementations maintain the counts
incrementally. I am keeping the readable version.

---

## What is still open

Nothing in the algorithm distinguishes a real morphological boundary from a
statistical coincidence. I do not yet know whether another family of tokenisers
does better on this point, nor how one would properly measure the difference.

And the question of fairness between languages is wide open: a tokeniser trained
mostly on English shreds Finnish or Turkish. I would like to be able to put a
number on that. I am not there yet.

---

## How I work

Four rules I set myself at the start, and intend to keep across the whole
repository.

**Nothing to download.** The corpus lives in the code. All of my Master's
notebooks began with a `wget` to a university server or a Google Drive mount.
Two years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in my submitted coursework are quoted, not erased.** Where a result I
handed in was wrong or incomplete, I say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what I
expected, I change the conclusion, not the experiment.

**The code is commented in French.** This is a repository meant to be read as
much as run.

---

*French version, and the one I wrote first: [README_FR.md](README_FR.md).*

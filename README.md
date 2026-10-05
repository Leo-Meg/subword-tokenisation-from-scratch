# Subword tokenisation, written by hand

Byte Pair Encoding implemented from scratch, with no dependencies, then put to the test.

**Léo Mégret**, MSc Computational Linguistics, Université Paris Cité

> **Repository status, version 2.** I am doing this work in stages, each in its
> own folder. I publish as I go rather than once everything is finished.

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

## Published versions

| | Folder | Contents | Tests |
|---|---|---|---:|
| **1** | `1.tokenisation_python_projet` | Byte Pair Encoding written by hand | 15 |
| **2** | `2.tokenisation_python_projet` | WordPiece, and why the merge criterion changes everything | 12 |

That is **27 tests** in total. Each folder contains everything the previous
one had, plus one step.

---

## Running the latest version

```bash
cd 2.tokenisation_python_projet
python -m src.wordpiece
python -m tests.test_wordpiece
```

---

## What is still open

I now have two algorithms and two vocabularies, and no way of saying which is
better. The segmentations differ, but I have nothing to decide whether that
difference matters.

What I am missing is measurements, and I do not yet know which ones would be
relevant.

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

---

*French version, which I wrote first, [README_FR.md](README_FR.md).*

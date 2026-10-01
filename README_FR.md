# Tokenisation en sous-mots, écrite à la main

Byte Pair Encoding implémenté depuis zéro, sans aucune dépendance, puis mis à l'épreuve.

**Léo Mégret** — Master Linguistique Informatique, Université Paris Cité

> **État du dépôt : version 1.** C'est la première étape d'un travail que je
> mène par étapes, chacune dans son propre dossier. Seule la version 1 existe à
> ce jour. Je publie au fur et à mesure plutôt qu'une fois tout terminé, parce
> que l'intérêt de ce travail est justement l'enchaînement des questions.

---

## Pourquoi ce dépôt

Un modèle de langue ne voit jamais de texte. Il voit une suite d'entiers. Tout
ce que le tokeniseur jette, le modèle ne pourra jamais le récupérer.

C'est l'objet le plus sous-estimé de toute la chaîne de traitement, et celui sur
lequel j'ai le plus appris en M1. Je le reprends ici en l'écrivant à la main,
parce qu'un tokeniseur appelé par une bibliothèque ne se comprend pas.

La question qui m'intéresse derrière l'algorithme est celle d'un linguiste :
ce système traite-t-il toutes les langues à égalité ?

---

## Ce qui existe aujourd'hui

### Version 1 — Byte Pair Encoding, écrit à la main

L'algorithme fondateur de la tokenisation moderne, celui de GPT, de RoBERTa et
de LLaMA.

| Fichier | Ce que j'y fais |
|---|---|
| `src/bpe.py` | `TokeniseurBPE` complet : pré-segmentation, comptage des paires, apprentissage des fusions, segmentation d'un mot par ordre de fusion, encodage et décodage inversibles, inspection du vocabulaire. |
| `tests/test_bpe.py` | 15 tests, dont l'inversibilité et le déterminisme. |

Origine universitaire : projet de M1 en binôme, implémenter BPE et WordPiece
depuis zéro, les entraîner sur des corpus français, comparer leurs vocabulaires
et leurs temps d'exécution.

---

## Lancer le code

```bash
cd 1.tokenisation_python_projet
python -m src.bpe
python -m tests.test_bpe
```

Aucune dépendance, pas même NumPy.

---

## Ce que je retiens de cette étape

**BPE n'est pas de la morphologie.** Il découpe volontiers `manger` en `mang` et
`er`, ce qui est juste, et `mer` en `m` et `er`, ce qui ne l'est pas du tout.
L'algorithme ne fait aucune différence entre les deux : il fusionne les paires
fréquentes, et rien d'autre.

**Un mot inconnu reste segmentable.** C'est la propriété qui fait tout l'intérêt
des sous-mots face à un vocabulaire de mots entiers, où le moindre mot jamais vu
devient `[UNK]`, un trou définitif. C'est aussi pour cela que GPT-2 travaille sur
des octets : avec 256 octets possibles, aucun texte n'est jamais hors-vocabulaire,
dans aucune langue et pour aucun émoji.

**Mon implémentation est naïve, et je l'assume.** Elle recompte toutes les paires
à chaque fusion, soit du O(V × M). Les implémentations sérieuses maintiennent les
comptes de façon incrémentale. Je garde la version lisible.

---

## Ce qui reste ouvert

Rien dans l'algorithme ne distingue une frontière morphologique réelle d'une
coïncidence statistique. Je ne sais pas encore si une autre famille de tokeniseurs
fait mieux sur ce point, ni comment on mesurerait proprement la différence.

Et la question de l'équité entre langues reste entière : un tokeniseur entraîné
majoritairement sur de l'anglais découpe le finnois ou le turc en charpie. Je
voudrais pouvoir le chiffrer, je n'en suis pas là.

---

## Comment je travaille

Quatre règles que je me suis données en commençant, et que je compte tenir sur
tout le dépôt.

**Rien à télécharger.** Le corpus est dans le code. Mes notebooks de master
commençaient tous par un `wget` vers un serveur universitaire ou un montage de
Google Drive ; deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à une
commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu en cours était faux ou incomplet, je le dis et je donne le résultat
correct.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, je change la conclusion, pas l'expérience.

**Le code est commenté en français.** C'est un dépôt à lire autant qu'à exécuter.

---

*Version anglaise : [README.md](README.md).*

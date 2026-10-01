# Tokenisation en sous-mots, écrite à la main

Byte Pair Encoding implémenté depuis zéro, sans aucune dépendance, puis mis à l'épreuve.

**Léo Mégret**, Master Linguistique Informatique, Université Paris Cité

> **État du dépôt, version 1.** C'est la première étape d'un travail que je mène
> par étapes, chacune dans son propre dossier. Seule la version 1 existe à ce
> jour. Je publie au fur et à mesure plutôt qu'une fois tout terminé.

---

## Pourquoi ce dépôt

Un modèle de langue ne reçoit pas du texte, il reçoit une suite d'entiers. Ce que
le tokeniseur jette, le modèle ne peut pas le récupérer ensuite.

C'est la partie de la chaîne de traitement sur laquelle j'ai le plus appris en
M1, et je la reprends ici en l'écrivant à la main. Un tokeniseur appelé depuis
une bibliothèque ne se comprend pas.

La question qui m'intéresse derrière l'algorithme est celle d'un linguiste. Ce
système traite-t-il toutes les langues à égalité.

---

## Ce qui existe aujourd'hui

### Version 1, Byte Pair Encoding écrit à la main

L'algorithme fondateur de la tokenisation moderne, celui de GPT, de RoBERTa et de
LLaMA.

| Fichier | Ce que j'y fais |
|---|---|
| `src/bpe.py` | `TokeniseurBPE` complet. Pré-segmentation, comptage des paires, apprentissage des fusions, segmentation d'un mot par ordre de fusion, encodage et décodage inversibles, inspection du vocabulaire. |
| `tests/test_bpe.py` | 15 tests, dont l'inversibilité et le déterminisme. |

Origine universitaire. Projet de M1 en binôme, implémenter BPE et WordPiece depuis
zéro, les entraîner sur des corpus français, comparer leurs vocabulaires et leurs
temps d'exécution.

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

**BPE ne fait pas de morphologie.** Il découpe `manger` en `mang` et `er`, ce qui
est juste, et `mer` en `m` et `er`, ce qui ne l'est pas. L'algorithme ne fait
aucune différence entre les deux cas, il fusionne les paires fréquentes.

**Un mot inconnu reste segmentable.** C'est ce qui rend les sous-mots utiles face
à un vocabulaire de mots entiers, où tout mot jamais vu devient `[UNK]`. C'est
aussi pour cela que GPT-2 travaille sur des octets. Avec 256 octets possibles,
aucun texte n'est hors-vocabulaire, dans aucune langue et pour aucun émoji.

**Mon implémentation est naïve et je l'assume.** Elle recompte toutes les paires à
chaque fusion, soit du O(V × M). Les implémentations sérieuses maintiennent les
comptes de façon incrémentale. Je garde la version lisible.

---

## Ce qui reste ouvert

Rien dans l'algorithme ne distingue une frontière morphologique réelle d'une
coïncidence statistique. Je ne sais pas si une autre famille de tokeniseurs fait
mieux sur ce point, ni comment on mesurerait la différence.

La question de l'équité entre langues reste entière. Un tokeniseur entraîné
majoritairement sur de l'anglais découpe le finnois ou le turc en morceaux très
courts. Je voudrais pouvoir le chiffrer, je n'en suis pas là.

---

## Comment je travaille

Quatre règles que je me suis données en commençant, et que je compte tenir sur
tout le dépôt.

**Rien à télécharger.** Le corpus est écrit dans le code. Mes notebooks de master
commençaient tous par un `wget` vers un serveur universitaire ou un montage de
Google Drive. Deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à une
commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu en cours était faux ou incomplet, je le dis et je donne le résultat
correct.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, j'écris ce que j'ai trouvé.

**Le code est commenté en français.**

---

*Version anglaise, [README.md](README.md).*

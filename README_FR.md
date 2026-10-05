# Tokenisation en sous-mots, écrite à la main

Byte Pair Encoding implémenté depuis zéro, sans aucune dépendance, puis mis à l'épreuve.

**Léo Mégret**, Master Linguistique Informatique, Université Paris Cité

> **État du dépôt, version 2.** Je mène ce travail par étapes, chacune dans son
> propre dossier. Je publie au fur et à mesure plutôt qu'une fois tout terminé.

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

## Les versions publiées

| | Dossier | Contenu | Tests |
|---|---|---|---:|
| **1** | `1.tokenisation_python_projet` | Byte Pair Encoding écrit à la main | 15 |
| **2** | `2.tokenisation_python_projet` | WordPiece, et pourquoi le critère de fusion change tout | 12 |

Soit **27 tests** au total. Chaque dossier contient tout le contenu du
précédent, plus une étape.

---

## Lancer la dernière version

```bash
cd 2.tokenisation_python_projet
python -m src.wordpiece
python -m tests.test_wordpiece
```

---

## Ce qui reste ouvert

J'ai maintenant deux algorithmes et deux vocabulaires, et aucun moyen de dire
lequel est le meilleur. Les segmentations diffèrent, mais je n'ai rien pour
décider si cette différence compte.

Il me manque des mesures, et je ne sais pas encore lesquelles seraient
pertinentes.

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

---

*Version anglaise, [README.md](README.md).*

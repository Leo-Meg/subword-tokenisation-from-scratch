# La tokenisation en sous-mots, écrite à la main

Byte Pair Encoding et WordPiece implémentés from scratch, sans aucune dépendance, puis mis à l'épreuve.

**Léo Mégret** — implémentation, notes techniques en français.  
**Haeeul Hwang** — documentation anglaise, documentation coréenne, relecture.

> **État du dépôt** — étape 2 sur 4, publiée par Léo Mégret.

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

## Les étapes

| | Étape | Modules | Tests | Publiée par | Note détaillée (français) |
|---|---|---|---:|---|---|
| **1** | Byte Pair Encoding écrit à la main | `src/bpe.py` | 15 | Léo | [`byte-pair-encoding`](notes/byte-pair-encoding.md) |
| **2** | WordPiece, et pourquoi le critère de fusion change tout | `src/wordpiece.py` | 12 | Léo | [`wordpiece`](notes/wordpiece.md) |

**27** tests au total.

---

## Ce que chaque étape a trouvé

**1. Byte Pair Encoding écrit à la main**  
Sans dictionnaire et sans annotation, BPE retrouve des morphèmes. Les premières fusions apprises sur un corpus français sont le `s` du pluriel, puis `e`, `es`, `en`, `ent`.

**2. WordPiece, et pourquoi le critère de fusion change tout**  
Même corpus, même taille de vocabulaire, une ligne de différence dans le critère : le recouvrement de Jaccard entre les deux vocabulaires est de 0,18.

---

## Lancer le code

```bash
python -m src.wordpiece

python -m tests.test_bpe
python -m tests.test_wordpiece
```

---

## Comment nous travaillons

**Rien à télécharger.** Le corpus est écrit dans le code. Tous mes
notebooks de master commençaient par un `wget` vers un serveur universitaire ou
un montage de Google Drive ; deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à
une commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu était faux ou incomplet, je le dis et je donne le bon.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, j'écris ce que j'ai trouvé.

**Le code est commenté en français**, parce que c'est la langue dans laquelle je
l'ai pensé. La documentation est en anglais, et en coréen quand Haeeul est
passée dessus.

---

*English: [README.md](README.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)*

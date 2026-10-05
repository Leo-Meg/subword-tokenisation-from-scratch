# Version 2, WordPiece, et pourquoi le critère de fusion change tout

> **Où j'en suis.** BPE fusionne la paire la plus *fréquente*. WordPiece fusionne
> la paire au meilleur *score de vraisemblance*. Cette version implémente le
> second, compare les deux, et tombe sur un résultat que je n'attendais pas.

---

## Nouveautés par rapport à la version 1

| Fichier | Ce que j'ajoute |
|---|---|
| `src/wordpiece.py` | `TokeniseurWordPiece` complet (critère de score, marquage `##`, segmentation par plus longue correspondance, encodage/décodage), et une fonction `comparer()` qui entraîne les deux algorithmes sur le même corpus au même budget. |
| `tests/test_wordpiece.py` | 12 tests, dont l'isolation de la différence de critère sur un exemple minimal. |

## Lancer le code

```bash
python -m src.wordpiece
```

```bash
python -m tests.test_wordpiece
```

---

## La différence, en une formule

```
BPE        : fusionner argmax  f(ab)
WordPiece  : fusionner argmax  f(ab) / ( f(a) × f(b) )
```

Le critère change de nature. Le dénominateur pénalise les paires dont les deux
éléments sont déjà fréquents séparément.

Un exemple qui m'a débloqué, en français, `e` et `s` sont tous deux
extrêmement fréquents. BPE fusionne `es` très tôt, simplement parce que la paire
apparaît souvent. WordPiece se demande. *`es` apparaît-il plus souvent que ce
qu'on attendrait si `e` et `s` étaient indépendants ?*

C'est exactement l'**information mutuelle ponctuelle** (PMI) qu'on utilise en
linguistique de corpus pour extraire des collocations. Je l'ai isolé dans
`test_le_score_penalise_les_symboles_deja_frequents`, sur un cas construit, BPE
trouve deux paires à égalité là où WordPiece en préfère nettement une.

### Et la convention de marquage

| | BPE | WordPiece |
|---|---|---|
| marqueur | `</w>` en **fin** de mot | `##` en **continuation** |
| `tokenisation` | `tokenis` `ati` `on</w>` | `tok` `##enisa` `##tion` |

C'est pour cela qu'on voit `##ing` dans le vocabulaire de BERT et jamais `ing`
tout court, le début de mot et la continuation ne se mélangent jamais.

---

## Le résultat que je n'attendais pas, et qui m'a le plus appris

Sur mon corpus de 25 phrases, à budget de vocabulaire égal (300 symboles).

| | fertilité (jetons/mot) |
|---|---:|
| BPE | **1,78** |
| WordPiece | **4,41** |

WordPiece est **deux fois et demie moins efficace**. Et ses unités apprises sont
des monstres.

```
'##orpholog', '##ocabula', '##rpholog', '##hograph', '##cabula', '##pholog'
```

**Ce n'est pas un bug de mon implémentation.** Le score `f(ab)/(f(a)·f(b))`
atteint son maximum, la valeur 1, pour une paire dont les deux éléments
n'apparaissent **que** ensemble, c'est-à-dire pour un hapax. C'est le biais bien
connu de la PMI vers les basses fréquences.

WordPiece dépense donc son budget de vocabulaire sur des séquences **rares mais
parfaitement corrélées** (`orpholog`, qui n'apparaît que dans *morphologie*), au
lieu des séquences fréquentes qui feraient baisser la fertilité.

Sur les milliards de tokens dont dispose BERT, l'effet disparaît, aucune
sous-chaîne utile n'est un hapax. Sur 25 phrases, il domine tout.

> **La leçon que je retiens :** un algorithme n'a pas de propriétés dans
> l'absolu, il en a **à une échelle donnée**. Je n'aurais jamais vu ça en
> appelant `BertTokenizer.from_pretrained(...)`.

Vérifié par `test_biais_de_la_pmi_vers_les_basses_frequences`, qui montre aussi
le maximum théorique atteint par un hapax construit exprès.

---

## Deux vocabulaires qui ne se recouvrent presque pas

En normalisant les marqueurs pour comparer les chaînes brutes.

```
BPE seul       : 197 symboles
commun         :  85
WordPiece seul : 184
indice de Jaccard : 0,182
```

Moins de 20 % de recouvrement. **Il n'existe pas de « bonne » segmentation en
sous-mots**, il existe des critères, et chacun favorise un type d'unité.

---

## Une différence pratique en faveur de BPE

Sur un mot inconnu.

- **BPE** dégrade caractère par caractère, le mot reste segmentable,
- **WordPiece** est tout ou rien, si un morceau ne passe pas, **le mot entier**
  devient `[UNK]`.

C'est le comportement réel de BERT, et c'est un vrai argument en faveur de BPE
dans les domaines à vocabulaire ouvert, médecine, chimie, noms propres.

Vérifié par `test_mot_impossible_devient_unk_en_entier`.

---

## Une note d'honnêteté sur mon implémentation

Le WordPiece original de Google (Schuster & Nakajima, 2012. Wu et al., 2016)
n'est pas exactement défini par ce rapport, il choisit la fusion qui maximise
la vraisemblance du corpus sous un modèle unigramme. Le rapport
`f(ab)/(f(a)·f(b))` en est l'approximation usuelle, celle qu'utilise la
documentation de Hugging Face et celle que nous avions implémentée en M1.

Je le mentionne parce que la différence compte pour interpréter mes résultats.
l'objectif exact atténue un peu le biais vers les basses fréquences, sans le
supprimer.

---

## Ce qui reste ouvert

J'ai maintenant deux algorithmes et deux vocabulaires, et aucun moyen de dire
lequel est le meilleur. Les segmentations diffèrent, mais je n'ai rien pour
décider si cette différence compte.

Il me manque des mesures, et je ne sais pas encore lesquelles seraient
pertinentes.

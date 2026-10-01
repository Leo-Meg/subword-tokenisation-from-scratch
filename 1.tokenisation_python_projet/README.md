# Version 1 — Byte Pair Encoding, écrit à la main

> **Où j'en suis.** Je commence par l'algorithme fondateur de la tokenisation
> moderne : celui de GPT, de RoBERTa, de LLaMA. Entraînement, encodage,
> décodage — tout est écrit et testé, sans dépendance.

---

## Ce que contient cette version

| Fichier | Ce que j'y fais |
|---|---|
| `src/bpe.py` | `TokeniseurBPE` complet : pré-segmentation, comptage des paires, apprentissage des fusions, segmentation d'un mot par ordre de fusion, encodage/décodage inversible, inspection du vocabulaire. |
| `tests/test_bpe.py` | 15 tests sans dépendance, dont l'inversibilité et le déterminisme. |

## Origine universitaire

Projet de M1 en binôme (« projet tower defense ») : implémenter BPE et
WordPiece from scratch, les entraîner sur des corpus français, comparer leurs
vocabulaires et leurs temps d'exécution.

C'est le travail dont je suis le plus content de ma première année, parce qu'il
porte sur l'objet le plus sous-estimé de toute la chaîne de traitement.

## Lancer le code

```bash
python -m src.bpe
```

```bash
python -m tests.test_bpe
```

Aucune dépendance — pas même NumPy.

---

## Pourquoi la tokenisation mérite un projet entier

Un modèle de langue ne voit **jamais** de texte. Il voit une suite d'entiers.
Tout ce que le tokeniseur jette, le modèle ne pourra jamais le récupérer.

Concrètement, le tokeniseur décide :

- **combien coûte une phrase** — donc le prix d'un appel d'API et la taille
  utile du contexte ;
- **si un mot rare est représentable** ou explosé en dix fragments ;
- **quelles langues sont avantagées** — et c'est le point qui m'intéresse le
  plus en tant que linguiste. Un tokeniseur entraîné majoritairement sur de
  l'anglais découpe le finnois ou le turc en charpie. C'est ce que je voudrais
  pouvoir mesurer, et je n'en suis pas encore là.

---

## L'algorithme, en trois lignes

1. partir de l'alphabet (un symbole = un caractère) ;
2. compter toutes les paires de symboles adjacents dans le corpus ;
3. fusionner la paire la plus fréquente, l'ajouter au vocabulaire, recommencer.

BPE vient de la **compression de données** (Gage, 1994) et a été détourné vers
la traduction automatique par Sennrich, Haddow & Birch (2016).

---

## Le résultat qui m'impressionne toujours

Sans aucune connaissance linguistique, sans dictionnaire, sans annotation, BPE
retrouve spontanément des **morphèmes**. Les 20 premières fusions apprises sur
mon corpus français :

```
's' + '</w>'   -> 's</w>'        (marque du pluriel !)
'e' + '</w>'   -> 'e</w>'
'e' + 's</w>'  -> 'es</w>'
'e' + 'n'      -> 'en'
'en' + 't'     -> 'ent'
...
'ent' + '</w>' -> 'ent</w>'      (désinence verbale de 3e pers. pluriel)
'm' + 'ent</w>' -> 'ment</w>'    (suffixe adverbial)
```

L'algorithme n'a jamais entendu parler de pluriel ni de suffixe adverbial. Ces
unités émergent **parce qu'elles sont fréquentes**. C'est de la morphologie
distributionnelle obtenue par simple comptage.

Et avec un vocabulaire un peu plus grand, les mots fréquents deviennent des
unités entières :

```
'développement</w>', 'apprentissage</w>', 'linguistique</w>', 'automatique</w>'
```

**Mais ce n'est pas de la morphologie correcte.** BPE découpera volontiers
`manger` en `mang` + `er`, ce qui est juste — et `mer` en `m` + `er`, ce qui ne
l'est pas du tout. Il ne fait aucune différence entre les deux, et je ne sais
pas encore quantifier à quel point cela coûte.

---

## Les trois pièges que je documente dans le code

### 1. Le marqueur de fin de mot n'est pas décoratif

Sans `</w>`, BPE ne peut pas distinguer un suffixe d'une séquence interne :
`ment` dans *vraiment* (final) et dans *mentir* (initial) recevraient le même
symbole. Le marqueur rend la **position** observable.

Il a une seconde fonction, tout aussi essentielle : c'est lui qui indique où
remettre les espaces au décodage. Un tokeniseur sans marqueur de frontière
n'est pas inversible — et l'inversibilité est indispensable dès qu'on génère.

### 2. L'ordre des fusions doit être respecté à l'encodage

**C'est le point que j'avais raté dans mon projet de M1.** Ma première version
cherchait, à chaque étape, la fusion applicable la plus fréquente. C'est faux :
cela produit des segmentations différentes de celles obtenues à l'entraînement.

La règle correcte : à chaque tour, appliquer la fusion de **plus petit rang**
(donc apprise le plus tôt) parmi celles qui sont applicables. C'est ce que fait
la bibliothèque `tokenizers` de Hugging Face, et c'est ce qui garantit qu'un
même mot est toujours segmenté de la même façon.

### 3. Il faut départager les égalités, sinon rien n'est reproductible

Quand deux paires ont la même fréquence, `max()` renvoie celle que l'ordre
d'itération du dictionnaire présente en premier — c'est-à-dire n'importe
laquelle. J'ajoute un critère lexicographique : arbitraire, mais **déterministe**.
Vérifié par `test_entrainement_deterministe`.

---

## Ce que les tests garantissent

| propriété | test |
|---|---|
| le tokeniseur est **inversible** | `test_aller_retour_encodage_decodage` |
| deux entraînements donnent le **même** modèle | `test_entrainement_deterministe` |
| un mot inconnu reste **segmentable** (pas `[UNK]` en bloc) | `test_mot_inconnu_reste_segmentable` |
| plus de vocabulaire = moins de jetons | `test_plus_de_fusions_donne_moins_de_jetons` |
| les mots fréquents coûtent **un seul** jeton | `test_mots_frequents_deviennent_un_seul_jeton` |

---

## Une limite que j'assume, et une que je signale

**Assumée** : mon implémentation recompte toutes les paires à chaque fusion,
soit du O(V × M). Les implémentations sérieuses maintiennent les comptes de
façon incrémentale. Je garde la version naïve parce qu'elle est lisible.
Mesurer le coût réel reste à faire.

**Signalée** : mon BPE travaille sur des **caractères**, donc un caractère
jamais vu à l'entraînement devient `[UNK]` — un trou définitif. C'est
précisément pour cela que GPT-2 travaille sur des **octets** : avec 256 octets
possibles, aucun texte n'est jamais hors-vocabulaire, en aucune langue et pour
aucun émoji.

---

## Ce qui reste ouvert

BPE fusionne les paires les plus fréquentes, et rien d'autre. Rien dans
l'algorithme ne distingue une frontière morphologique réelle d'une coïncidence
statistique, et c'est pourquoi `mer` devient `m` + `er`.

Je ne sais pas encore si une autre famille de tokeniseurs fait mieux sur ce
point, ni comment on mesurerait proprement la différence.

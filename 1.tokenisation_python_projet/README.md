# Version 1, Byte Pair Encoding écrit à la main

> **Où j'en suis.** Je commence par l'algorithme fondateur de la tokenisation
> moderne, celui de GPT, de RoBERTa et de LLaMA. Entraînement, encodage et
> décodage, tout est écrit et testé, sans dépendance.

---

## Ce que contient cette version

| Fichier | Ce que j'y fais |
|---|---|
| `src/bpe.py` | `TokeniseurBPE` complet. Pré-segmentation, comptage des paires, apprentissage des fusions, segmentation d'un mot par ordre de fusion, encodage et décodage inversibles, inspection du vocabulaire. |
| `tests/test_bpe.py` | 15 tests sans dépendance, dont l'inversibilité et le déterminisme. |

## Origine universitaire

Projet de M1 en binôme, le « projet tower defense », qui demandait d'implémenter
BPE et WordPiece depuis zéro, de les entraîner sur des corpus français, et de
comparer leurs vocabulaires et leurs temps d'exécution.

C'est le travail dont je suis le plus content de ma première année, parce qu'il
porte sur l'objet le moins regardé de la chaîne de traitement.

## Lancer le code

```bash
python -m src.bpe
```

```bash
python -m tests.test_bpe
```

Aucune dépendance, pas même NumPy.

---

## Pourquoi la tokenisation mérite un projet entier

Un modèle de langue ne reçoit pas du texte, il reçoit une suite d'entiers. Ce que
le tokeniseur jette, le modèle ne peut pas le récupérer ensuite.

Concrètement, le tokeniseur décide trois choses.

- **Combien coûte une phrase**, donc le prix d'un appel d'API et la taille utile
  du contexte.
- **Si un mot rare reste représentable** ou se trouve éclaté en dix fragments.
- **Quelles langues sont avantagées**, et c'est le point qui m'intéresse le plus
  en tant que linguiste. Un tokeniseur entraîné majoritairement sur de l'anglais
  découpe le finnois ou le turc en morceaux très courts. C'est ce que je voudrais
  pouvoir mesurer, et je n'en suis pas encore là.

---

## L'algorithme, en trois lignes

1. Partir de l'alphabet, un symbole pour un caractère.
2. Compter toutes les paires de symboles adjacents dans le corpus.
3. Fusionner la paire la plus fréquente, l'ajouter au vocabulaire, recommencer.

BPE vient de la compression de données (Gage, 1994) et a été détourné vers la
traduction automatique par Sennrich, Haddow et Birch (2016).

---

## Le résultat qui m'a surpris

Sans aucune connaissance linguistique, sans dictionnaire et sans annotation, BPE
retrouve des morphèmes. Voici les 20 premières fusions apprises sur mon corpus
français.

```
's' + '</w>'   -> 's</w>'        (marque du pluriel)
'e' + '</w>'   -> 'e</w>'
'e' + 's</w>'  -> 'es</w>'
'e' + 'n'      -> 'en'
'en' + 't'     -> 'ent'
...
'ent' + '</w>' -> 'ent</w>'      (désinence verbale de 3e pers. pluriel)
'm' + 'ent</w>' -> 'ment</w>'    (suffixe adverbial)
```

L'algorithme n'a jamais entendu parler de pluriel ni de suffixe adverbial. Ces
unités apparaissent parce qu'elles sont fréquentes. C'est de la morphologie
distributionnelle obtenue par simple comptage.

Avec un vocabulaire un peu plus grand, les mots fréquents deviennent des unités
entières.

```
'développement</w>', 'apprentissage</w>', 'linguistique</w>', 'automatique</w>'
```

**Mais ce n'est pas de la morphologie correcte.** BPE découpe `manger` en `mang`
et `er`, ce qui est juste, et `mer` en `m` et `er`, ce qui ne l'est pas. Il ne
fait aucune différence entre les deux cas, et je ne sais pas encore quantifier ce
que cela coûte.

---

## Les trois pièges que je documente dans le code

### 1. Le marqueur de fin de mot n'est pas décoratif

Sans `</w>`, BPE ne peut pas distinguer un suffixe d'une séquence interne. Le
`ment` de *vraiment*, qui est final, et celui de *mentir*, qui est initial,
recevraient le même symbole. Le marqueur rend la position observable.

Il a une seconde fonction. C'est lui qui indique où remettre les espaces au
décodage. Un tokeniseur sans marqueur de frontière n'est pas inversible, et
l'inversibilité est indispensable dès qu'on génère.

### 2. L'ordre des fusions doit être respecté à l'encodage

C'est le point que j'avais raté dans mon projet de M1. Ma première version
cherchait, à chaque étape, la fusion applicable la plus fréquente. C'est faux,
cela produit des segmentations différentes de celles obtenues à l'entraînement.

La règle correcte est d'appliquer, à chaque tour, la fusion de plus petit rang,
donc apprise le plus tôt, parmi celles qui sont applicables. C'est ce que fait la
bibliothèque `tokenizers` de Hugging Face, et c'est ce qui garantit qu'un même
mot est toujours segmenté de la même façon.

### 3. Il faut départager les égalités, sinon rien n'est reproductible

Quand deux paires ont la même fréquence, `max()` renvoie celle que l'ordre
d'itération du dictionnaire présente en premier, c'est-à-dire n'importe laquelle.
J'ajoute un critère lexicographique, arbitraire mais déterministe. Vérifié par
`test_entrainement_deterministe`.

---

## Ce que les tests garantissent

| propriété | test |
|---|---|
| le tokeniseur est inversible | `test_aller_retour_encodage_decodage` |
| deux entraînements donnent le même modèle | `test_entrainement_deterministe` |
| un mot inconnu reste segmentable, pas `[UNK]` en bloc | `test_mot_inconnu_reste_segmentable` |
| plus de vocabulaire donne moins de jetons | `test_plus_de_fusions_donne_moins_de_jetons` |
| les mots fréquents coûtent un seul jeton | `test_mots_frequents_deviennent_un_seul_jeton` |

---

## Une limite que j'assume, et une que je signale

**Assumée.** Mon implémentation recompte toutes les paires à chaque fusion, soit
du O(V × M). Les implémentations sérieuses maintiennent les comptes de façon
incrémentale. Je garde la version naïve parce qu'elle est lisible. Mesurer le
coût réel reste à faire.

**Signalée.** Mon BPE travaille sur des caractères, donc un caractère jamais vu à
l'entraînement devient `[UNK]`, et l'information est perdue. C'est pour cela que
GPT-2 travaille sur des octets. Avec 256 octets possibles, aucun texte n'est
hors-vocabulaire, en aucune langue et pour aucun émoji.

---

## Ce qui reste ouvert

BPE fusionne les paires les plus fréquentes, et rien d'autre. Rien dans
l'algorithme ne distingue une frontière morphologique réelle d'une coïncidence
statistique, et c'est pourquoi `mer` devient `m` et `er`.

Je ne sais pas encore si une autre famille de tokeniseurs fait mieux sur ce point,
ni comment on mesurerait proprement la différence.

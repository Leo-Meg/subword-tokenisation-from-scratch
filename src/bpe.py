"""
Byte Pair Encoding (BPE), implémenté à la main.

D'où ça vient
-------------
Projet de M1 (« projet tower defense », en binôme) : nous devions implémenter
BPE et WordPiece from scratch, les entraîner sur des corpus français, puis
comparer leurs vocabulaires et leurs temps d'exécution. C'est le travail dont je
suis le plus content de toute ma première année, parce qu'il porte sur l'objet
le plus sous-estimé de la chaîne : le tokeniseur.

Pourquoi la tokenisation mérite un projet à elle seule
-------------------------------------------------------
Un modèle de langue ne voit **jamais** de texte. Il voit une suite d'entiers.
Tout ce que le tokeniseur jette, le modèle ne pourra jamais le récupérer ; tout
ce que le tokeniseur découpe mal, le modèle devra le réapprendre.

Concrètement, le tokeniseur décide :
* combien coûte une phrase (donc le prix d'un appel d'API, et la taille du
  contexte utile) ;
* si un mot rare est représentable ou explosé en dix fragments ;
* et — c'est le point qui m'intéresse le plus en tant que linguiste — quelles
  langues sont avantagées. Un tokeniseur entraîné majoritairement sur de
  l'anglais découpe le finnois ou le turc en charpie.

L'idée de BPE
-------------
BPE vient de la compression de données (Gage, 1994) et a été détourné vers la
traduction automatique par Sennrich, Haddow & Birch (2016). L'algorithme tient
en trois lignes :

  1. partir de l'alphabet (un symbole = un caractère) ;
  2. compter toutes les paires de symboles adjacents dans le corpus ;
  3. fusionner la paire la plus fréquente, l'ajouter au vocabulaire, recommencer.

Le résultat est remarquable : sans aucune connaissance linguistique, sans
dictionnaire, sans annotation, BPE retrouve spontanément des morphèmes. Les
suffixes fréquents (« -ment », « -tion », « -ait ») émergent parce qu'ils sont
fréquents. C'est de la morphologie distributionnelle, obtenue par comptage.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

# Marqueur de fin de mot. Sans lui, BPE ne peut pas distinguer un suffixe d'une
# séquence interne : « ment » dans « vraiment » (final) et dans « mentir »
# (initial) recevraient le même symbole. Le marqueur rend la position observable.
FIN_DE_MOT = "</w>"


class TokeniseurBPE:
    """Byte Pair Encoding entraînable, avec encodage et décodage.

    Attributs après entraînement :
        vocabulaire: liste ordonnée des symboles (l'ordre reflète l'ancienneté).
        fusions: liste ordonnée des règles ``(a, b)``. **L'ordre est capital** :
            à l'encodage, il faut réappliquer les fusions dans l'ordre exact où
            elles ont été apprises, sinon on n'obtient pas la même segmentation.
    """

    def __init__(self, jetons_speciaux: list[str] | None = None):
        self.jetons_speciaux = jetons_speciaux or ["[PAD]", "[UNK]"]
        self.vocabulaire: list[str] = []
        self.fusions: list[tuple[str, str]] = []
        self.rang_fusion: dict[tuple[str, str], int] = {}
        self._cache_mots: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ #
    # Prétraitement
    # ------------------------------------------------------------------ #

    @staticmethod
    def pre_segmenter(texte: str) -> list[str]:
        """Découpe un texte en « mots » avant l'application de BPE.

        Cette étape s'appelle la **pré-segmentation** (pre-tokenization) et elle
        est bien plus déterminante qu'il n'y paraît : elle interdit à BPE de
        fusionner par-dessus les frontières qu'elle pose. Si je sépare la
        ponctuation ici, aucune fusion ne pourra jamais produire « chat, ».

        Je sépare la ponctuation et je découpe sur les espaces. C'est une
        version simplifiée de ce que fait GPT-2 (qui utilise une expression
        régulière beaucoup plus fine, attachant notamment l'espace au mot
        suivant).
        """
        texte = re.sub(r"([.,!?;:()\"«»…])", r" \1 ", texte)
        return texte.lower().split()

    @staticmethod
    def compter_mots(corpus: list[str]) -> dict[str, int]:
        """Fréquence de chaque mot du corpus.

        Astuce essentielle de BPE : on ne travaille **jamais** sur le texte
        brut, mais sur le dictionnaire des mots pondérés par leur fréquence.
        Un corpus de 10 millions de tokens peut ne contenir que 100 000 mots
        distincts : on divise le travail par cent.
        """
        compteur: Counter[str] = Counter()
        for texte in corpus:
            compteur.update(TokeniseurBPE.pre_segmenter(texte))
        return dict(compteur)

    # ------------------------------------------------------------------ #
    # Entraînement
    # ------------------------------------------------------------------ #

    @staticmethod
    def compter_paires(
        decoupages: dict[str, list[str]], frequences: dict[str, int]
    ) -> dict[tuple[str, str], int]:
        """Compte les paires de symboles adjacents, pondérées par la fréquence du mot."""
        paires: defaultdict[tuple[str, str], int] = defaultdict(int)
        for mot, symboles in decoupages.items():
            frequence = frequences[mot]
            for i in range(len(symboles) - 1):
                paires[(symboles[i], symboles[i + 1])] += frequence
        return dict(paires)

    @staticmethod
    def fusionner(
        paire: tuple[str, str], decoupages: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Applique une fusion à tous les découpages."""
        a, b = paire
        fusionne = a + b
        resultat = {}
        for mot, symboles in decoupages.items():
            nouveaux, i = [], 0
            while i < len(symboles):
                if i < len(symboles) - 1 and symboles[i] == a and symboles[i + 1] == b:
                    nouveaux.append(fusionne)
                    i += 2
                else:
                    nouveaux.append(symboles[i])
                    i += 1
            resultat[mot] = nouveaux
        return resultat

    def entrainer(self, corpus: list[str], taille_vocabulaire: int = 1000,
                  verbeux: bool = False) -> None:
        """Apprend le vocabulaire et les règles de fusion.

        Args:
            taille_vocabulaire: taille visée, **jetons spéciaux et alphabet
                compris**. Le nombre de fusions apprises vaut donc
                ``taille_vocabulaire − |alphabet| − |jetons spéciaux|``.

        Note sur la complexité : ma version recompte toutes les paires à chaque
        fusion, ce qui donne du O(V × M) où M est le nombre de mots distincts.
        Les implémentations sérieuses (`tokenizers` de Hugging Face) maintiennent
        les comptes de façon incrémentale et n'invalident que les mots touchés
        par la fusion. Je garde la version naïve parce qu'elle est lisible, et je
        mesure le coût réel dans le module `evaluation`.
        """
        frequences = self.compter_mots(corpus)

        # Chaque mot devient une liste de caractères + le marqueur de fin.
        decoupages = {
            mot: list(mot) + [FIN_DE_MOT] for mot in frequences
        }

        alphabet = sorted({c for mot in frequences for c in mot})
        self.vocabulaire = list(self.jetons_speciaux) + alphabet + [FIN_DE_MOT]
        self.fusions = []

        while len(self.vocabulaire) < taille_vocabulaire:
            paires = self.compter_paires(decoupages, frequences)
            if not paires:
                break  # plus rien à fusionner : tous les mots sont un seul symbole

            # `max` avec une clé composite : fréquence d'abord, puis ordre
            # lexicographique. Ce second critère est arbitraire mais **nécessaire**
            # pour que l'entraînement soit déterministe en cas d'égalité.
            meilleure = max(paires.items(), key=lambda kv: (kv[1], kv[0]))[0]

            decoupages = self.fusionner(meilleure, decoupages)
            self.fusions.append(meilleure)
            self.vocabulaire.append(meilleure[0] + meilleure[1])

            if verbeux and len(self.fusions) % 100 == 0:
                print(f"  {len(self.fusions)} fusions | dernière : "
                      f"{meilleure[0]!r} + {meilleure[1]!r} "
                      f"-> {meilleure[0] + meilleure[1]!r}")

        self.rang_fusion = {paire: i for i, paire in enumerate(self.fusions)}
        self._cache_mots = {}

    # ------------------------------------------------------------------ #
    # Encodage
    # ------------------------------------------------------------------ #

    def _segmenter_mot(self, mot: str) -> list[str]:
        """Segmente un mot en réappliquant les fusions **dans l'ordre appris**.

        C'est le point que j'avais raté dans mon projet de M1. Ma première
        version cherchait à chaque étape la fusion applicable la plus fréquente,
        ce qui donne des segmentations différentes de celles de l'entraînement.
        La règle correcte : à chaque tour, appliquer la fusion de **plus petit
        rang** (donc apprise le plus tôt) parmi celles qui sont applicables.

        C'est ce que fait `tokenizers` de Hugging Face, et c'est aussi ce qui
        garantit qu'un même mot est toujours segmenté de la même façon.
        """
        if mot in self._cache_mots:
            return self._cache_mots[mot]

        symboles = list(mot) + [FIN_DE_MOT]

        while len(symboles) > 1:
            candidates = [
                (self.rang_fusion[(symboles[i], symboles[i + 1])], i)
                for i in range(len(symboles) - 1)
                if (symboles[i], symboles[i + 1]) in self.rang_fusion
            ]
            if not candidates:
                break
            _, position = min(candidates)
            symboles = (symboles[:position]
                        + [symboles[position] + symboles[position + 1]]
                        + symboles[position + 2:])

        self._cache_mots[mot] = symboles
        return symboles

    def tokeniser(self, texte: str) -> list[str]:
        """Texte -> liste de sous-mots.

        Les caractères jamais vus à l'entraînement sont remplacés par ``[UNK]``.
        C'est la faiblesse du BPE « au caractère » : un émoji ou un idéogramme
        inconnu devient un trou. C'est précisément pour cela que GPT-2 travaille
        sur des **octets** et non des caractères : avec 256 octets possibles,
        aucun texte n'est jamais hors-vocabulaire.
        """
        connus = set(self.vocabulaire)
        jetons = []
        for mot in self.pre_segmenter(texte):
            for symbole in self._segmenter_mot(mot):
                jetons.append(symbole if symbole in connus else "[UNK]")
        return jetons

    def encoder(self, texte: str) -> list[int]:
        """Texte -> liste d'identifiants entiers."""
        index = {symbole: i for i, symbole in enumerate(self.vocabulaire)}
        inconnu = index.get("[UNK]", 0)
        return [index.get(j, inconnu) for j in self.tokeniser(texte)]

    def decoder(self, identifiants: list[int]) -> str:
        """Identifiants -> texte.

        La reconstruction repose entièrement sur ``</w>`` : c'est lui qui
        indique où remettre les espaces. Un tokeniseur sans marqueur de frontière
        n'est pas inversible, et l'inversibilité est indispensable dès qu'on
        génère du texte.
        """
        symboles = [self.vocabulaire[i] for i in identifiants
                    if 0 <= i < len(self.vocabulaire)]
        texte = "".join(symboles).replace(FIN_DE_MOT, " ")
        return texte.strip()

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #

    def premieres_fusions(self, n: int = 20) -> list[str]:
        """Les ``n`` premières fusions, sous forme lisible."""
        return [f"{a!r} + {b!r} -> {a + b!r}" for a, b in self.fusions[:n]]

    def sous_mots_les_plus_longs(self, n: int = 15) -> list[str]:
        """Les plus longues unités apprises — souvent des mots entiers fréquents."""
        appris = [s for s in self.vocabulaire
                  if s not in self.jetons_speciaux and len(s) > 1]
        return sorted(appris, key=len, reverse=True)[:n]


# --------------------------------------------------------------------------- #
# Corpus de démonstration
# --------------------------------------------------------------------------- #

CORPUS_FRANCAIS = [
    "le développement rapide des modèles de langue transforme le traitement automatique des langues",
    "la tokenisation détermine ce que le modèle peut représenter et ce qu'il ne pourra jamais apprendre",
    "un modèle de langue ne voit jamais de texte, il voit uniquement une suite d'entiers",
    "les linguistes s'intéressent au fonctionnement interne des représentations apprises",
    "le traitement automatique des langues combine linguistique et apprentissage automatique",
    "les étudiants du master travaillent sur des corpus annotés manuellement",
    "l'apprentissage automatique nécessite énormément de données annotées",
    "la segmentation en sous-mots permet de traiter les mots inconnus efficacement",
    "les modèles multilingues partagent un vocabulaire entre toutes les langues",
    "chaque fusion apprise augmente la taille du vocabulaire d'une unité",
    "la fréquence des séquences détermine entièrement l'ordre des fusions",
    "le développement des systèmes automatiques transforme rapidement la discipline",
    "les représentations vectorielles encodent des régularités distributionnelles",
    "un vocabulaire trop petit produit des séquences très longues",
    "un vocabulaire trop grand produit des représentations mal estimées",
    "la morphologie flexionnelle du français complique la segmentation",
    "les langues agglutinantes posent des difficultés particulières aux tokeniseurs",
    "on observe que les suffixes fréquents émergent spontanément des fusions",
    "le comptage des paires adjacentes constitue le cœur de l'algorithme",
    "l'algorithme fusionne itérativement la paire la plus fréquente du corpus",
    "les traitements automatiques modernes reposent sur des architectures profondes",
    "la linguistique informatique étudie le langage avec des méthodes formelles",
    "les modèles apprennent des régularités sans supervision explicite",
    "chaque langue possède une morphologie et une orthographe spécifiques",
    "le modèle apprend automatiquement à découper les mots en unités utiles",
]


if __name__ == "__main__":
    print("=== Byte Pair Encoding, entraîné à la main ===\n")

    tokeniseur = TokeniseurBPE()
    tokeniseur.entrainer(CORPUS_FRANCAIS, taille_vocabulaire=300)

    print(f"corpus       : {len(CORPUS_FRANCAIS)} phrases")
    print(f"mots uniques : {len(tokeniseur.compter_mots(CORPUS_FRANCAIS))}")
    print(f"vocabulaire  : {len(tokeniseur.vocabulaire)} symboles")
    print(f"fusions      : {len(tokeniseur.fusions)}\n")

    print("Les 20 premières fusions apprises :")
    for ligne in tokeniseur.premieres_fusions(20):
        print(f"  {ligne}")

    print("\nLes plus longues unités apprises :")
    print("  " + ", ".join(repr(s) for s in tokeniseur.sous_mots_les_plus_longs(12)))

    print("\n=== Segmentation ===\n")
    for phrase in [
        "le développement automatique",
        "les linguistes travaillent",
        "un mot totalement inconnu : anticonstitutionnellement",
    ]:
        jetons = tokeniseur.tokeniser(phrase)
        print(f"  « {phrase} »")
        print(f"    -> {jetons}")
        print(f"    -> {len(jetons)} jetons pour {len(phrase.split())} mots\n")

    print("=== Aller-retour encodage / décodage ===\n")
    for phrase in ["le modèle apprend", "la tokenisation détermine tout"]:
        identifiants = tokeniseur.encoder(phrase)
        reconstruit = tokeniseur.decoder(identifiants)
        print(f"  original    : {phrase}")
        print(f"  identifiants: {identifiants[:12]}{'…' if len(identifiants) > 12 else ''}")
        print(f"  reconstruit : {reconstruit}")
        print(f"  identique   : {reconstruit == phrase}\n")

    print(
        "Ce qu'il faut remarquer\n"
        "-----------------------\n"
        "Les premières fusions sont des bigrammes de caractères très fréquents\n"
        "(« en », « es », « le »…). Puis, sans qu'on lui ait rien dit de la\n"
        "morphologie du français, l'algorithme construit des unités qui\n"
        "*ressemblent* à des morphèmes : des suffixes, des préfixes, des mots\n"
        "grammaticaux entiers.\n"
        "\n"
        "C'est de la morphologie distributionnelle obtenue par simple comptage.\n"
        "Elle n'est pas linguistiquement correcte — BPE découpera volontiers\n"
        "« manger » en « mang » + « er » mais aussi « mer » en « m » + « er » —\n"
        "et c'est justement ce que j'analyse dans les versions suivantes."
    )

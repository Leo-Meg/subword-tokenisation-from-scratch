"""
WordPiece — le tokeniseur de BERT, implémenté à la main.

D'où ça vient
-------------
Même projet de M1 que BPE (« projet tower defense ») : nous devions implémenter
les deux et comparer. C'est cette comparaison qui m'a fait comprendre que
« tokeniseur en sous-mots » ne désigne pas une seule méthode mais une famille,
et que le choix du **critère de fusion** change tout.

La différence avec BPE, en une formule
---------------------------------------
BPE fusionne la paire la plus **fréquente**. WordPiece fusionne la paire au
meilleur **score** :

    score(a, b) = fréquence(ab) / (fréquence(a) × fréquence(b))

Ce n'est pas un détail de réglage, c'est un changement de nature. Le
dénominateur pénalise les paires dont les deux éléments sont *déjà* très
fréquents séparément.

Un exemple qui m'a débloqué : en français, « e » et « s » sont tous deux
extrêmement fréquents. BPE fusionne « es » très tôt, simplement parce que la
paire apparaît souvent. WordPiece se demande : « es » apparaît-il *plus souvent
que ce qu'on attendrait si e et s étaient indépendants ? » Si non, la fusion
n'apporte aucune information et il ne la fait pas.

C'est exactement l'**information mutuelle ponctuelle** (PMI) qu'on utilise en
linguistique de corpus pour extraire des collocations. WordPiece cherche des
séquences *statistiquement liées*, pas seulement fréquentes — d'où le nom donné
par ses auteurs : la fusion qui maximise la vraisemblance du corpus sous un
modèle unigramme.

La seconde différence : la convention de marquage
--------------------------------------------------
BPE marque la **fin** des mots (``</w>``). WordPiece marque les **continuations**
avec ``##`` : « tokenisation » -> ``token``, ``##isa``, ``##tion``.

Conséquence pratique : le début de mot est distingué de la suite, donc les
sous-mots initiaux et internes ne se mélangent jamais dans le vocabulaire. C'est
pour ça qu'on voit ``##ing`` dans le vocabulaire de BERT et jamais ``ing`` tout
court.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .bpe import TokeniseurBPE

PREFIXE_CONTINUATION = "##"


class TokeniseurWordPiece:
    """WordPiece entraînable, avec segmentation par plus longue correspondance.

    Deux différences majeures avec mon `TokeniseurBPE` :

      * le critère de fusion (score de vraisemblance, pas fréquence brute) ;
      * la segmentation à l'inférence — WordPiece n'a **pas besoin** de mémoriser
        les fusions, il applique un algorithme glouton de plus longue
        correspondance de gauche à droite sur le vocabulaire final.
    """

    def __init__(self, jetons_speciaux: list[str] | None = None):
        self.jetons_speciaux = jetons_speciaux or [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
        ]
        self.vocabulaire: list[str] = []
        self._vocabulaire_set: set[str] = set()
        self._cache_mots: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ #
    # Entraînement
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decouper_initialement(mot: str) -> list[str]:
        """« chat » -> ``['c', '##h', '##a', '##t']``."""
        return [mot[0]] + [PREFIXE_CONTINUATION + c for c in mot[1:]]

    @staticmethod
    def _recoller(a: str, b: str) -> str:
        """Fusionne deux symboles en retirant le ``##`` du second.

        ``('to', '##ken')`` -> ``'token'`` ; ``('##ken', '##isa')`` -> ``'##kenisa'``.
        Le préfixe du **premier** symbole est conservé : c'est lui qui porte
        l'information « début de mot ou continuation ».
        """
        return a + (b[len(PREFIXE_CONTINUATION):]
                    if b.startswith(PREFIXE_CONTINUATION) else b)

    @staticmethod
    def calculer_scores(
        decoupages: dict[str, list[str]], frequences: dict[str, int]
    ) -> dict[tuple[str, str], float]:
        """``score(a, b) = f(ab) / (f(a) × f(b))``.

        C'est le cœur de WordPiece. Je calcule en une passe les fréquences des
        symboles isolés et celles des paires, puis je forme le rapport.
        """
        frequences_paires: defaultdict[tuple[str, str], int] = defaultdict(int)
        frequences_symboles: defaultdict[str, int] = defaultdict(int)

        for mot, symboles in decoupages.items():
            frequence = frequences[mot]
            if len(symboles) == 1:
                frequences_symboles[symboles[0]] += frequence
                continue
            for i in range(len(symboles) - 1):
                frequences_symboles[symboles[i]] += frequence
                frequences_paires[(symboles[i], symboles[i + 1])] += frequence
            frequences_symboles[symboles[-1]] += frequence

        return {
            paire: frequence / (frequences_symboles[paire[0]]
                                * frequences_symboles[paire[1]])
            for paire, frequence in frequences_paires.items()
        }

    def entrainer(self, corpus: list[str], taille_vocabulaire: int = 1000,
                  verbeux: bool = False) -> None:
        """Apprend le vocabulaire.

        Contrairement à BPE, on ne conserve **pas** la liste des fusions : le
        vocabulaire final suffit, puisque la segmentation se fera par plus
        longue correspondance.
        """
        frequences = TokeniseurBPE.compter_mots(corpus)
        decoupages = {mot: self._decouper_initialement(mot)
                      for mot in frequences if mot}

        alphabet: set[str] = set()
        for mot in frequences:
            if not mot:
                continue
            alphabet.add(mot[0])
            alphabet.update(PREFIXE_CONTINUATION + c for c in mot[1:])

        self.vocabulaire = list(self.jetons_speciaux) + sorted(alphabet)

        while len(self.vocabulaire) < taille_vocabulaire:
            scores = self.calculer_scores(decoupages, frequences)
            if not scores:
                break

            meilleure = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
            nouveau = self._recoller(*meilleure)

            a, b = meilleure
            for mot, symboles in decoupages.items():
                nouveaux, i = [], 0
                while i < len(symboles):
                    if i < len(symboles) - 1 and symboles[i] == a and symboles[i + 1] == b:
                        nouveaux.append(nouveau)
                        i += 2
                    else:
                        nouveaux.append(symboles[i])
                        i += 1
                decoupages[mot] = nouveaux

            if nouveau not in self.vocabulaire:
                self.vocabulaire.append(nouveau)

            if verbeux and len(self.vocabulaire) % 100 == 0:
                print(f"  {len(self.vocabulaire)} symboles | dernier : {nouveau!r}")

        self._vocabulaire_set = set(self.vocabulaire)
        self._cache_mots = {}

    # ------------------------------------------------------------------ #
    # Segmentation
    # ------------------------------------------------------------------ #

    def segmenter_mot(self, mot: str) -> list[str]:
        """Plus longue correspondance, de gauche à droite (*greedy longest-match*).

        À chaque étape, on cherche le plus long préfixe présent dans le
        vocabulaire, on l'émet, et on recommence sur le reste en préfixant par
        ``##``.

        Le comportement en cas d'échec est celui de BERT et il est brutal : si à
        un moment aucun préfixe ne convient, **le mot entier** devient `[UNK]`,
        et pas seulement le morceau problématique. C'est une différence notable
        avec BPE, qui dégrade caractère par caractère.

        L'algorithme est **glouton** et donc pas optimal : il peut produire une
        segmentation plus longue que nécessaire. C'est le compromis assumé de
        WordPiece — Unigram/SentencePiece, lui, cherche la segmentation la plus
        probable par programmation dynamique.
        """
        if mot in self._cache_mots:
            return self._cache_mots[mot]

        jetons: list[str] = []
        reste = mot
        debut_de_mot = True

        while reste:
            fin = len(reste)
            trouve = None
            while fin > 0:
                candidat = reste[:fin]
                if not debut_de_mot:
                    candidat = PREFIXE_CONTINUATION + candidat
                if candidat in self._vocabulaire_set:
                    trouve = candidat
                    break
                fin -= 1

            if trouve is None:
                self._cache_mots[mot] = ["[UNK]"]
                return ["[UNK]"]

            jetons.append(trouve)
            reste = reste[fin:]
            debut_de_mot = False

        self._cache_mots[mot] = jetons
        return jetons

    def tokeniser(self, texte: str) -> list[str]:
        jetons = []
        for mot in TokeniseurBPE.pre_segmenter(texte):
            jetons.extend(self.segmenter_mot(mot))
        return jetons

    def encoder(self, texte: str) -> list[int]:
        index = {s: i for i, s in enumerate(self.vocabulaire)}
        inconnu = index.get("[UNK]", 0)
        return [index.get(j, inconnu) for j in self.tokeniser(texte)]

    def decoder(self, identifiants: list[int]) -> str:
        """Reconstruction : on recolle les ``##`` et on sépare le reste par un espace."""
        mots: list[str] = []
        for i in identifiants:
            if not 0 <= i < len(self.vocabulaire):
                continue
            symbole = self.vocabulaire[i]
            if symbole.startswith(PREFIXE_CONTINUATION) and mots:
                mots[-1] += symbole[len(PREFIXE_CONTINUATION):]
            else:
                mots.append(symbole)
        return " ".join(mots)

    def continuations(self, n: int = 15) -> list[str]:
        """Les sous-mots de continuation les plus longs — les « suffixes » appris."""
        candidats = [s for s in self.vocabulaire
                     if s.startswith(PREFIXE_CONTINUATION) and len(s) > 4]
        return sorted(candidats, key=len, reverse=True)[:n]


# --------------------------------------------------------------------------- #
# Comparaison des deux algorithmes
# --------------------------------------------------------------------------- #


def comparer(corpus: list[str], taille_vocabulaire: int = 300,
             phrases_test: list[str] | None = None) -> dict:
    """Entraîne BPE et WordPiece sur le même corpus et compare leurs sorties."""
    bpe = TokeniseurBPE()
    bpe.entrainer(corpus, taille_vocabulaire=taille_vocabulaire)

    wp = TokeniseurWordPiece()
    wp.entrainer(corpus, taille_vocabulaire=taille_vocabulaire)

    phrases_test = phrases_test or corpus[:10]
    jetons_bpe = sum(len(bpe.tokeniser(p)) for p in phrases_test)
    jetons_wp = sum(len(wp.tokeniser(p)) for p in phrases_test)
    nb_mots = sum(len(TokeniseurBPE.pre_segmenter(p)) for p in phrases_test)

    return {
        "bpe": bpe,
        "wordpiece": wp,
        "jetons_bpe": jetons_bpe,
        "jetons_wordpiece": jetons_wp,
        "nb_mots": nb_mots,
        "fertilite_bpe": jetons_bpe / nb_mots,
        "fertilite_wordpiece": jetons_wp / nb_mots,
    }


if __name__ == "__main__":
    from .bpe import CORPUS_FRANCAIS

    print("=== WordPiece — le tokeniseur de BERT ===\n")

    wp = TokeniseurWordPiece()
    wp.entrainer(CORPUS_FRANCAIS, taille_vocabulaire=300)

    print(f"vocabulaire : {len(wp.vocabulaire)} symboles")
    print(f"  dont continuations (##) : "
          f"{sum(s.startswith(PREFIXE_CONTINUATION) for s in wp.vocabulaire)}\n")

    print("Continuations les plus longues apprises :")
    print("  " + ", ".join(repr(s) for s in wp.continuations(12)))

    print("\n=== BPE contre WordPiece, même corpus, même budget ===\n")
    resultats = comparer(CORPUS_FRANCAIS, taille_vocabulaire=300)
    bpe = resultats["bpe"]

    print(f"  fertilité BPE       : {resultats['fertilite_bpe']:.3f} jetons/mot")
    print(f"  fertilité WordPiece : {resultats['fertilite_wordpiece']:.3f} jetons/mot")
    print("  (la fertilité mesure le coût moyen d'un mot ; plus bas = mieux)\n")

    print("Segmentations comparées :\n")
    mots = ["développement", "linguistique", "automatiquement",
            "tokenisation", "anticonstitutionnellement", "chat"]
    largeur = max(len(m) for m in mots)
    for mot in mots:
        s_bpe = bpe.tokeniser(mot)
        s_wp = wp.tokeniser(mot)
        print(f"  {mot:<{largeur}}")
        print(f"    BPE       ({len(s_bpe)}) : {s_bpe}")
        print(f"    WordPiece ({len(s_wp)}) : {s_wp}")
    print()

    print("Recouvrement des vocabulaires :")
    # On normalise en retirant les marqueurs, pour comparer les *chaînes*.
    from .bpe import FIN_DE_MOT
    nu_bpe = {s.replace(FIN_DE_MOT, "") for s in bpe.vocabulaire}
    nu_wp = {s.replace(PREFIXE_CONTINUATION, "") for s in wp.vocabulaire}
    commun = nu_bpe & nu_wp
    print(f"  BPE seul       : {len(nu_bpe - nu_wp)}")
    print(f"  commun         : {len(commun)}")
    print(f"  WordPiece seul : {len(nu_wp - nu_bpe)}")
    print(f"  indice de Jaccard : {len(commun) / len(nu_bpe | nu_wp):.3f}")

    print(
        "\nCe que révèle la comparaison\n"
        "----------------------------\n"
        "* BPE fusionne ce qui est FRÉQUENT. WordPiece fusionne ce qui est\n"
        "  statistiquement LIÉ (le dénominateur f(a)×f(b) pénalise les paires\n"
        "  dont les éléments sont déjà courants séparément). C'est exactement\n"
        "  l'information mutuelle ponctuelle utilisée en linguistique de corpus\n"
        "  pour extraire des collocations.\n"
        "\n"
        "* Les deux vocabulaires se recouvrent à peine (Jaccard ≈ 0,18). Il\n"
        "  n'existe pas de « bonne » segmentation en sous-mots : il existe des\n"
        "  critères, et chacun favorise un type d'unité.\n"
        "\n"
        "* LE RÉSULTAT QUI M'A LE PLUS APPRIS. Sur ce corpus minuscule, WordPiece\n"
        "  a une fertilité BIEN PIRE que BPE, et ses unités apprises sont des\n"
        "  monstres comme '##orpholog' ou '##hograph'. Ce n'est pas un bug : le\n"
        "  score f(ab)/(f(a)·f(b)) atteint son maximum, 1, pour une paire dont\n"
        "  les deux éléments n'apparaissent QUE ensemble — c'est-à-dire pour un\n"
        "  hapax. C'est le biais bien connu de la PMI vers les basses fréquences.\n"
        "\n"
        "  WordPiece dépense donc son budget de vocabulaire sur des séquences\n"
        "  rares mais parfaitement corrélées, au lieu des séquences fréquentes\n"
        "  qui feraient baisser la fertilité. Sur les milliards de tokens dont\n"
        "  dispose BERT, l'effet disparaît (aucune sous-chaîne n'est un hapax).\n"
        "  Sur 25 phrases, il domine tout.\n"
        "\n"
        "  Leçon que je retiens : un algorithme n'a pas de propriétés dans\n"
        "  l'absolu, il en a À UNE ÉCHELLE DONNÉE. Je n'aurais jamais vu ça en\n"
        "  appelant `BertTokenizer.from_pretrained(...)`.\n"
        "\n"
        "* Sur un mot inconnu, WordPiece est aussi plus brutal : si un morceau ne\n"
        "  passe pas, le mot ENTIER devient [UNK]. BPE, lui, dégrade caractère\n"
        "  par caractère. C'est un argument pratique en faveur de BPE pour les\n"
        "  domaines à vocabulaire ouvert (médecine, chimie, noms propres)."
    )

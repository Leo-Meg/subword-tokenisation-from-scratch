"""
Tests de la version 2 — WordPiece et comparaison avec BPE.

    python -m tests.test_wordpiece
"""

from __future__ import annotations

from src.bpe import CORPUS_FRANCAIS, TokeniseurBPE
from src.wordpiece import PREFIXE_CONTINUATION, TokeniseurWordPiece, comparer


def _wp(taille: int = 300) -> TokeniseurWordPiece:
    t = TokeniseurWordPiece()
    t.entrainer(CORPUS_FRANCAIS, taille_vocabulaire=taille)
    return t


def test_decoupage_initial_marque_les_continuations() -> None:
    assert TokeniseurWordPiece._decouper_initialement("chat") == [
        "c", "##h", "##a", "##t"
    ]


def test_recollage_retire_le_prefixe_du_second() -> None:
    assert TokeniseurWordPiece._recoller("to", "##ken") == "token"
    assert TokeniseurWordPiece._recoller("##ken", "##isa") == "##kenisa"


def test_score_est_bien_un_rapport_de_vraisemblance() -> None:
    """f(ab) / (f(a) × f(b)) — vérifié à la main sur un cas minuscule."""
    decoupages = {"ab": ["a", "b"], "ac": ["a", "c"]}
    frequences = {"ab": 1, "ac": 1}
    scores = TokeniseurWordPiece.calculer_scores(decoupages, frequences)
    # f(a) = 2 (dans les deux mots), f(b) = 1, f(ab) = 1 -> 1 / (2 × 1) = 0.5
    assert abs(scores[("a", "b")] - 0.5) < 1e-12
    assert abs(scores[("a", "c")] - 0.5) < 1e-12


def test_le_score_penalise_les_symboles_deja_frequents() -> None:
    """La différence de nature avec BPE, isolée sur un exemple.

    « ab » et « cd » apparaissent aussi souvent l'un que l'autre, mais « a » est
    beaucoup plus fréquent globalement. WordPiece doit donc préférer « cd ».
    """
    decoupages = {
        "ab": ["a", "b"],
        "ax": ["a", "x"],
        "ay": ["a", "y"],
        "cd": ["c", "d"],
    }
    frequences = {"ab": 5, "ax": 5, "ay": 5, "cd": 5}
    scores = TokeniseurWordPiece.calculer_scores(decoupages, frequences)
    assert scores[("c", "d")] > scores[("a", "b")]

    # BPE, lui, les trouve à égalité (même fréquence de paire).
    paires = TokeniseurBPE.compter_paires(decoupages, frequences)
    assert paires[("a", "b")] == paires[("c", "d")]


def test_vocabulaire_contient_des_continuations() -> None:
    wp = _wp()
    continuations = [s for s in wp.vocabulaire
                     if s.startswith(PREFIXE_CONTINUATION)]
    assert len(continuations) > 10


def test_segmentation_commence_sans_prefixe() -> None:
    """Le premier jeton d'un mot n'a jamais de ``##``, les suivants toujours."""
    wp = _wp()
    jetons = wp.segmenter_mot("linguistique")
    assert not jetons[0].startswith(PREFIXE_CONTINUATION)
    assert all(j.startswith(PREFIXE_CONTINUATION) for j in jetons[1:])


def test_plus_longue_correspondance() -> None:
    """Vérifie que la segmentation est bien gloutonne, du plus long au plus court."""
    wp = TokeniseurWordPiece()
    wp.vocabulaire = ["[UNK]", "t", "to", "tok", "##e", "##en", "##ken"]
    wp._vocabulaire_set = set(wp.vocabulaire)
    # « token » : le plus long préfixe disponible est « tok », puis « ##en ».
    assert wp.segmenter_mot("token") == ["tok", "##en"]


def test_mot_impossible_devient_unk_en_entier() -> None:
    """Le comportement de BERT : c'est tout ou rien."""
    wp = TokeniseurWordPiece()
    wp.vocabulaire = ["[UNK]", "a", "##b"]
    wp._vocabulaire_set = set(wp.vocabulaire)
    assert wp.segmenter_mot("azz") == ["[UNK]"]
    assert wp.segmenter_mot("ab") == ["a", "##b"]


def test_aller_retour_encodage_decodage() -> None:
    wp = _wp(500)
    for phrase in ["le modele apprend", "la tokenisation determine tout"]:
        assert wp.decoder(wp.encoder(phrase)) == phrase


def test_entrainement_deterministe() -> None:
    assert _wp(250).vocabulaire == _wp(250).vocabulaire


def test_les_deux_vocabulaires_different() -> None:
    """Le point central : deux critères de fusion, deux vocabulaires."""
    from src.bpe import FIN_DE_MOT

    resultats = comparer(CORPUS_FRANCAIS, taille_vocabulaire=300)
    nu_bpe = {s.replace(FIN_DE_MOT, "") for s in resultats["bpe"].vocabulaire}
    nu_wp = {s.replace(PREFIXE_CONTINUATION, "")
             for s in resultats["wordpiece"].vocabulaire}
    jaccard = len(nu_bpe & nu_wp) / len(nu_bpe | nu_wp)
    assert 0.05 < jaccard < 0.6, f"recouvrement inattendu : {jaccard:.3f}"


def test_biais_de_la_pmi_vers_les_basses_frequences() -> None:
    """Le résultat que je documente comme la leçon principale de cette version.

    Le score f(ab)/(f(a)·f(b)) vaut 1 — son maximum — pour une paire dont les
    deux éléments n'apparaissent QUE ensemble. WordPiece dépense donc son budget
    sur des séquences rares mais parfaitement corrélées, ce qui dégrade la
    fertilité sur un petit corpus.
    """
    resultats = comparer(CORPUS_FRANCAIS, taille_vocabulaire=300)
    assert resultats["fertilite_wordpiece"] > resultats["fertilite_bpe"], (
        "sur ce corpus minuscule, WordPiece doit être MOINS efficace que BPE"
    )

    # Le maximum théorique du score est bien atteint par un hapax.
    decoupages = {"zq": ["z", "q"], "aa": ["a", "a"], "ab": ["a", "b"]}
    frequences = {"zq": 1, "aa": 50, "ab": 50}
    scores = TokeniseurWordPiece.calculer_scores(decoupages, frequences)
    assert scores[("z", "q")] == 1.0
    assert scores[("z", "q")] > scores[("a", "b")]


def executer_tous_les_tests() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    echecs = 0
    for test in tests:
        try:
            test()
            print(f"  [OK]     {test.__name__}")
        except AssertionError as e:
            echecs += 1
            print(f"  [ÉCHEC]  {test.__name__} : {e}")
    print(f"\n{len(tests) - echecs}/{len(tests)} tests passés.")
    if echecs:
        raise SystemExit(1)


if __name__ == "__main__":
    print("=== Tests — version 2 : WordPiece ===\n")
    executer_tous_les_tests()

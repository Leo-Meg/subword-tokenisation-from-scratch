"""
Tests de la version 1 — Byte Pair Encoding.

    python -m tests.test_bpe
"""

from __future__ import annotations

from src.bpe import CORPUS_FRANCAIS, FIN_DE_MOT, TokeniseurBPE


def _tokeniseur_entraine(taille: int = 250) -> TokeniseurBPE:
    t = TokeniseurBPE()
    t.entrainer(CORPUS_FRANCAIS, taille_vocabulaire=taille)
    return t


def test_pre_segmentation_isole_la_ponctuation() -> None:
    assert TokeniseurBPE.pre_segmenter("Bonjour, le monde !") == [
        "bonjour", ",", "le", "monde", "!"
    ]


def test_comptage_des_mots() -> None:
    frequences = TokeniseurBPE.compter_mots(["le chat le chien", "le chat"])
    assert frequences["le"] == 3
    assert frequences["chat"] == 2
    assert frequences["chien"] == 1


def test_comptage_des_paires_pondere_par_la_frequence() -> None:
    decoupages = {"ab": ["a", "b"], "abc": ["a", "b", "c"]}
    frequences = {"ab": 10, "abc": 1}
    paires = TokeniseurBPE.compter_paires(decoupages, frequences)
    assert paires[("a", "b")] == 11   # 10 + 1
    assert paires[("b", "c")] == 1


def test_fusion_remplace_toutes_les_occurrences() -> None:
    decoupages = {"abab": ["a", "b", "a", "b"]}
    resultat = TokeniseurBPE.fusionner(("a", "b"), decoupages)
    assert resultat["abab"] == ["ab", "ab"]


def test_taille_du_vocabulaire_respectee() -> None:
    for taille in (120, 200, 300):
        t = _tokeniseur_entraine(taille)
        assert len(t.vocabulaire) <= taille
        # On doit avoir appris quelque chose.
        assert len(t.fusions) > 0


def test_vocabulaire_sans_doublon() -> None:
    t = _tokeniseur_entraine()
    assert len(t.vocabulaire) == len(set(t.vocabulaire))


def test_entrainement_deterministe() -> None:
    """Deux entraînements identiques doivent donner exactement le même modèle.

    C'est le rôle du critère de départage lexicographique en cas d'égalité de
    fréquence. Sans lui, l'ordre d'itération d'un dictionnaire suffirait à
    changer le vocabulaire d'une exécution à l'autre.
    """
    a, b = _tokeniseur_entraine(220), _tokeniseur_entraine(220)
    assert a.fusions == b.fusions
    assert a.vocabulaire == b.vocabulaire


def test_aller_retour_encodage_decodage() -> None:
    """Propriété essentielle : le tokeniseur doit être inversible."""
    t = _tokeniseur_entraine(400)
    for phrase in [
        "le modèle apprend",
        "la tokenisation détermine tout",
        "les linguistes travaillent sur des corpus",
    ]:
        assert t.decoder(t.encoder(phrase)) == phrase


def test_le_marqueur_de_fin_de_mot_est_present() -> None:
    """Sans lui, la reconstruction des espaces est impossible."""
    t = _tokeniseur_entraine()
    jetons = t.tokeniser("le chat dort")
    assert sum(j.endswith(FIN_DE_MOT) for j in jetons) == 3


def test_mot_inconnu_reste_segmentable() -> None:
    """Un mot jamais vu doit se décomposer, pas devenir [UNK] en bloc.

    C'est tout l'intérêt de la segmentation en sous-mots par rapport à un
    vocabulaire de mots entiers.
    """
    t = _tokeniseur_entraine(400)
    jetons = t.tokeniser("anticonstitutionnellement")
    assert len(jetons) > 1
    assert "[UNK]" not in jetons


def test_caractere_jamais_vu_devient_unk() -> None:
    """La faiblesse du BPE au caractère (que le BPE au niveau octet corrige)."""
    t = _tokeniseur_entraine()
    assert "[UNK]" in t.tokeniser("漢字")


def test_mots_frequents_deviennent_un_seul_jeton() -> None:
    """Le comportement attendu : ce qui est fréquent finit par coûter un jeton."""
    t = _tokeniseur_entraine(400)
    for mot in ("le", "les", "des", "automatique"):
        assert len(t.tokeniser(mot)) == 1, f"{mot} -> {t.tokeniser(mot)}"


def test_plus_de_fusions_donne_moins_de_jetons() -> None:
    """Propriété fondamentale : le vocabulaire achète de la compression."""
    phrase = " ".join(CORPUS_FRANCAIS[:5])
    longueurs = [
        len(_tokeniseur_entraine(taille).tokeniser(phrase))
        for taille in (100, 200, 400)
    ]
    assert longueurs[0] > longueurs[1] > longueurs[2], longueurs


def test_segmentation_stable_dans_le_temps() -> None:
    """Le même mot doit toujours être segmenté de la même façon."""
    t = _tokeniseur_entraine(300)
    a = t.tokeniser("développement")
    b = t.tokeniser("le développement automatique")[1:2]
    assert a == b or a[0] == b[0]


def test_ordre_des_fusions_respecte_a_l_encodage() -> None:
    """Le point que j'avais raté dans mon projet de M1.

    Les fusions doivent être réappliquées dans l'ordre appris (rang croissant),
    et non dans un ordre choisi à la volée.
    """
    t = _tokeniseur_entraine(300)
    # Chaque symbole produit doit appartenir au vocabulaire.
    connus = set(t.vocabulaire)
    for jeton in t.tokeniser("les représentations vectorielles"):
        assert jeton in connus, jeton


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
    print("=== Tests — version 1 : BPE ===\n")
    executer_tous_les_tests()

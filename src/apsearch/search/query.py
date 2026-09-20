"""Query construction for Greek legal text.

Two defects in the Greek morphological handling of full-text engines have to be
worked around, and both are corpus-correctness issues rather than tuning
preferences.

1. Unstable stemming across inflections
---------------------------------------
    αδικοπραξία -> αδικοπραξ     αδικοπραξίας -> αδικοπραξι
    αναίρεση    -> αναιρεσ       αναιρεσείων  -> αναιρεσει

A nominative query therefore misses genitive documents -- and the genitive is
the *normal* form in Greek legal prose ("λόγος αναιρέσεως", "αγωγή
αδικοπραξίας"). Since the unstable part is always a suffix, the shorter stem is
a prefix of the longer one, so emitting prefix matches (`αδικοπραξ*`) unifies
them. Only for stems of >= MIN_PREFIX_LEN characters: `δικ*` would match
δικαστήριο / δικαίωμα / δίκη and is worse than useless.

2. No effective stopword list
-----------------------------
The engine's default tokenizer yields `τ, την, κα, στ` for "του την και στο" --
these appear in 100% of documents and bloat the index. Filtering is done on
**surface forms, before stemming**, because after stemming the information
needed to do it safely is already gone:

    από (preposition)          -> απ  |  ΑΠ  (Άρειος Πάγος)        -> απ
    αν  (conjunction "if")     -> αν  |  ΑΝ  (Αναγκαστικός Νόμος)  -> αν

Dropping the stem ``απ`` would silently delete "Άρειος Πάγος" from queries. So
an all-caps short token is always treated as an acronym and never as a
stopword. This also keeps ΑΚ, ΠΚ, ΚΠολΔ, ΚΠΔ, ΝΔ, ΕΣΔΑ searchable.
"""

from __future__ import annotations

import re
import unicodedata

#: Below this, a prefix match is too broad to be useful.
MIN_PREFIX_LEN = 5

#: Greek vowels. The stemmer's instability is a single trailing vowel
#: (αδικοπραξ / αδικοπραξι), so trimming one -- when the stem stays long enough
#: to remain selective -- makes the prefix match symmetric in both directions.
_VOWELS = "αεηιουω"

#: Greek function words, accent-folded surface forms.
STOPWORDS: frozenset[str] = frozenset(
    """
    ο η το οι τα του τησ των τον την τουσ τισ
    ενα μια ενασ μιασ ενοσ
    και κι ειτε αλλα ομωσ ωστε οτι που πωσ ποια ποιο ποιοσ
    να θα ασ δεν μη μην αν εαν οταν αφου ενω καθωσ επειδη διοτι γιατι
    με σε για προσ κατα μετα πριν χωρισ παρα περι υπο επι εν εκ ωσ απο
    στο στη στον στην στουσ στισ στα στησ στου
    αυτο αυτη αυτοσ αυτα αυτων αυτου αυτησ
    ειναι ηταν εχει ειχε εχουν εχω ειχαν ηθελε
    οπωσ οπου οσο οσα οσοι ουτε μονο πολυ πιο
    καθε αλλο αλλη αλλοι ολα ολεσ ολοι ολο
    τοτε τωρα εδω εκει ναι οχι
    """.split()
)

#: An all-caps token of this length or shorter is treated as an acronym
#: (ΑΚ, ΑΠ, ΠΚ, ΝΔ, ΚΠΔ, ΚΠολΔ, ΕΣΔΑ) and never filtered as a stopword.
MAX_ACRONYM_LEN = 6

#: Signals that the user wants exact-phrase/operator semantics (quotes, OR, etc).
_SYNTAX_RE = re.compile(r'["\u201c\u201d]|(?<!\w)-\w|\bOR\b|\bAND\b')

_TOKEN_RE = re.compile(r"[\w\u0370-\u03ff\u1f00-\u1fff]+", re.UNICODE)


def fold(s: str) -> str:
    """Lowercase, strip diacritics, normalise final sigma."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", s).lower().replace("ς", "σ")


def has_operators(query: str) -> bool:
    return bool(_SYNTAX_RE.search(query or ""))


def is_acronym(token: str) -> bool:
    """All-caps (or mixed-caps like ΚΠολΔ) short token -> a legal abbreviation."""
    if len(token) > MAX_ACRONYM_LEN or len(token) < 2:
        return False
    letters = [c for c in token if c.isalpha()]
    if not letters:
        return False
    # Fully upper, or the common Greek style of caps with a lowercase infix.
    return token.isupper() or (letters[0].isupper() and sum(c.isupper() for c in letters) >= 2)


def content_tokens(query: str) -> list[str]:
    """Drop function words, keeping acronyms and anything numeric."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(query or ""):
        if any(c.isdigit() for c in tok) or is_acronym(tok):
            out.append(tok)
            continue
        if fold(tok) in STOPWORDS:
            continue
        out.append(tok)
    return out


def trim_unstable_suffix(stem: str) -> str:
    """Drop one trailing vowel so both stem variants share the prefix.

    ``αδικοπραξι`` -> ``αδικοπραξ``, which as a prefix now matches documents
    stemmed either way. Applied only while the result stays at least
    MIN_PREFIX_LEN characters, so selectivity is preserved.
    """
    if len(stem) > MIN_PREFIX_LEN and stem[-1] in _VOWELS:
        return stem[:-1]
    return stem

"""Every labelled game, by name: the DFL matches (dfl.py), Metrica's sample games (metrica.py),
SkillCorner's Clasicos (skillcorner.py) and whatever of PFF's World Cup is unpacked in the cache
(pff.py). None of them is the provided game."""

from possession import config, dfl, metrica, pff, skillcorner

# Params
DFL = tuple(sorted(dfl.MATCHES))
METRICA = tuple(sorted(metrica.MATCHES))
SKILLCORNER = tuple(sorted(skillcorner.MATCHES))
PFF = pff.available()
ALL = DFL + METRICA + SKILLCORNER + PFF


def default():
    """The games train() and evaluate() use unless told otherwise."""
    return DFL + (METRICA if config.METRICA_GAMES else ())


def _source(name):
    if name in skillcorner.MATCHES:
        return skillcorner
    if name.startswith(pff.PREFIX):
        return pff
    return metrica if name in metrica.MATCHES else dfl


def load(name):
    return _source(name).load(name)


def download(name):
    return _source(name).download(name)

from postpyc import guvectorize
from postyp import Array, Int32


def _modify(value: Int32, modifier: Int32) -> Int32:
    return (value * modifier + 2047) // 4096


def _ordinary_stat(
    base: Int32,
    iv: Int32,
    ev: Int32,
    level: Int32,
    nature: Int32,
) -> Int32:
    stat: Int32 = ((2 * base + iv + ev // 4) * level) // 100 + 5
    return (stat * nature) // 100


@guvectorize([], "(p),()->()")
def damage_batch(params: Array[Int32], roll: Int32, out: Array[Int32]) -> None:
    attacker_level: Int32 = params[0]
    attack: Int32 = _ordinary_stat(
        params[3],
        params[4],
        params[5],
        attacker_level,
        params[6],
    )
    attack = _modify(attack, params[12])
    defense: Int32 = _ordinary_stat(
        params[7],
        params[8],
        params[9],
        params[1],
        params[10],
    )
    defense = _modify(defense, params[11])

    result: Int32 = (((2 * attacker_level) // 5 + 2) * params[2] * attack) // defense
    result = result // 50 + 2
    result = (result * (100 - roll)) // 100
    result = _modify(result, params[13])

    type_mod: Int32 = params[14]
    factor: Int32 = 1
    if type_mod >= 0:
        for _index in range(type_mod):
            factor *= 2
        result *= factor
    else:
        for _index in range(-type_mod):
            factor *= 2
        result //= factor

    result = _modify(result, params[15])
    result = _modify(result, params[16])
    if result < 1:
        result = 1
    out[0] = result

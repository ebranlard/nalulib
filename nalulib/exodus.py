
from nalulib.exodus_info import exo_info
from nalulib.exodus_airfoil_layers import exo_layers
from nalulib.exodus_hex2quads import exo_hex2quads, exo_flatten
from nalulib.exodus_quads2hex import exo_quads2hex, exo_zextrude
from nalulib.exodus_rotate import exo_rotate_CLI, exo_rotate


def exo_get_times(filename):
    with ExodusIIFile(filename, mode="r") as exo:
        times = exo.get_times()
    return times

from nalulib.exodus_info import exo_info, exo_info_CLI
from nalulib.exodus_airfoil_layers import exo_layers, exo_layers_CLI
from nalulib.exodus_hex2quads import exo_flatten , exo_flatten_CLI
from nalulib.exodus_quads2hex import exo_zextrude, exo_zextrude_CLI
from nalulib.exodus_rotate import exo_rotate, exo_rotate_CLI


def exo_get_times(filename):
    with ExodusIIFile(filename, mode="r") as exo:
        times = exo.get_times()
    return times
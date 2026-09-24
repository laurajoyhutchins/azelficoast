"""The stable player seam used by the battle harness."""

from poke_env.player import SimpleHeuristicsPlayer


class AzelficoastPlayer(SimpleHeuristicsPlayer):
    """Temporary baseline behind the Azelficoast player interface.

    The harness depends on this class rather than a particular model or search
    implementation. Replacing the decision machinery should not require
    changing local, ladder, or challenge orchestration.
    """

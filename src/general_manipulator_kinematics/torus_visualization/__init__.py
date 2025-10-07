import importlib.util

if importlib.util.find_spec("matplotlib"):
    from .__so2_2_torus import SO22TorusPlotter
    from .__so2_3_torus import plot_so2_3_torus

    __all__ = ["SO22TorusPlotter", "plot_so2_3_torus"]

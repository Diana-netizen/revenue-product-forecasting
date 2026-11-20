"""Model registry. Imports are lazy so the LightGBM-only path never needs torch."""
import importlib

_MODULES = {"LightGBM": "lgbm", "GRU": "gru", "TFT": "tft"}


def get_registry(cfg, names=None):
    registry = {}
    for name in (names or cfg["models"]):
        m = importlib.import_module(f".{_MODULES[name]}", __package__)
        registry[name] = dict(
            sample=m.sample, default=m.DEFAULT, fit=m.fit, predict=m.predict, save=m.save,
            tune_targets=m.TUNE_TARGETS,
            max_steps=cfg["tuning"]["max_steps"][name], n_configs=cfg["tuning"]["n_configs"][name])
    return registry
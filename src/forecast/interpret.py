
"""TFT interpretability: variable-selection weights and attention maps, per product."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .constants import HORIZON, LOOKBACK
from .models.tft import predict_loader


def _get(raw, name):
    """Read a field across pytorch-forecasting versions."""
    try:
        return raw[name]
    except (KeyError, TypeError, IndexError, AttributeError):
        return getattr(raw, name, None)


def _to_numpy(value):
    """Convert tensors or array-like values to NumPy."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def interpret_tft(art, fd, target="Revenue"):
    """Extract TFT variable-selection weights and optional attention."""
    model, ds = art[target]

    res = model.predict(
        predict_loader(fd, ds),
        mode="raw",
        return_x=True,
        return_index=True
    )

    if hasattr(res, "output"):
        raw, index = res.output, res.index
    else:
        raw, _, index = res

    prods = index["Product"].astype(str).tolist()
    B = len(prods)

    # Variable-selection weights
    static = _to_numpy(_get(raw, "static_variables"))
    enc = _to_numpy(_get(raw, "encoder_variables"))
    dec = _to_numpy(_get(raw, "decoder_variables"))

    if static is None or enc is None or dec is None:
        raise ValueError(
            "TFT output does not contain all required variable-selection "
            "weights: static_variables, encoder_variables and decoder_variables."
        )

    static = static.reshape(B, -1)
    enc = enc.reshape(B, enc.shape[1], -1)
    dec = dec.reshape(B, dec.shape[1], -1)

    out = {
        "static": pd.DataFrame(
            static,
            index=prods,
            columns=model.static_variables
        ),
        "encoder": pd.DataFrame(
            enc.mean(axis=1),
            index=prods,
            columns=model.encoder_variables
        ),
        "decoder": pd.DataFrame(
            dec.mean(axis=1),
            index=prods,
            columns=model.decoder_variables
        )
    }

    out["decoder_by_horizon"] = {
        h: pd.DataFrame(
            dec[:, mask].mean(axis=1),
            index=prods,
            columns=model.decoder_variables
        )
        for h, mask in fd.horizon_masks.items()
        if np.asarray(mask).any()
    }

    # Attention weights are optional because their availability depends
    # on the PyTorch Forecasting version and model output configuration.
    att = _get(raw, "attention")

    if att is None:
        enc_att = _get(raw, "encoder_attention")
        dec_att = _get(raw, "decoder_attention")

        if enc_att is not None and dec_att is not None:
            enc_att = _to_numpy(enc_att)
            dec_att = _to_numpy(dec_att)
            att = np.concatenate([enc_att, dec_att], axis=-1)

    if att is not None:
        att = _to_numpy(att)

        if att.ndim >= 4:
            att = att.mean(axis=2)

        out["attention"] = dict(zip(prods, att))
        print(f"Attention shape: {att.shape}")
    else:
        out["attention"] = {}
        print(
            f"Attention weights unavailable for {target}. "
            "Variable-selection interpretation will still be generated."
        )

    return out


def top_drivers(interp, k=5):
    """Summarise the top variable-selection drivers for each product."""
    rows = []

    fmt = lambda s: ", ".join(
        f"{v} ({w:.2f})" for v, w in s.nlargest(k).items()
    )

    for p in interp["encoder"].index:
        rows.append({
            "Product": p,
            "Block": "History (encoder)",
            "Top": fmt(interp["encoder"].loc[p])
        })

        for h, df in interp["decoder_by_horizon"].items():
            rows.append({
                "Product": p,
                "Block": f"Known future - {h}",
                "Top": fmt(df.loc[p])
            })

    return pd.DataFrame(rows)


def plot_drivers(out, target, path, top=8):
    """Plot static, historical and known-future variable-selection weights."""
    prods = list(out["encoder"].index)

    fig, axes = plt.subplots(
        len(prods),
        3,
        figsize=(18, 3.4 * len(prods)),
        squeeze=False
    )

    for r, p in enumerate(prods):
        out["encoder"].loc[p].nlargest(top)[::-1].plot.barh(
            ax=axes[r, 0],
            color="steelblue"
        )
        axes[r, 0].set_title(
            f"{p} - {target}: encoder (history) weights"
        )

        hz = pd.DataFrame({
            h: df.loc[p]
            for h, df in out["decoder_by_horizon"].items()
        })

        if not hz.empty:
            selected = hz.loc[
                hz.mean(axis=1).nlargest(top).index
            ][::-1]
            selected.plot.barh(ax=axes[r, 1])

        axes[r, 1].set_title("Known-future weights by horizon")

        out["static"].loc[p].plot.barh(
            ax=axes[r, 2],
            color="seagreen"
        )
        axes[r, 2].set_title("Static weights")

    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_attention(out, fd, target, directory):
    """Plot attention maps only when attention weights are available."""
    if not out.get("attention"):
        print(
            f"Skipping attention plots for {target}: "
            "attention weights are unavailable."
        )
        return

    directory = Path(directory)

    for p, att in out["attention"].items():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(16, 4.5),
            gridspec_kw={"width_ratios": [1.6, 1]}
        )

        im = axes[0].imshow(
            att,
            aspect="auto",
            cmap="viridis",
            extent=[-LOOKBACK, HORIZON, HORIZON, 0]
        )

        axes[0].set_xlabel(
            "Attended position (days relative to origin)"
        )
        axes[0].set_ylabel("Forecast day")
        axes[0].set_title(f"{p} - {target}: attention map")
        plt.colorbar(im, ax=axes[0])

        for h, mask in fd.horizon_masks.items():
            mask = np.asarray(mask)

            if mask.any():
                axes[1].plot(
                    np.arange(-LOOKBACK, 0),
                    att[mask][:, :LOOKBACK].mean(axis=0),
                    label=h
                )

        axes[1].set_xlabel("Lookback day")
        axes[1].set_ylabel("Mean attention")
        axes[1].legend()
        axes[1].set_title("Attention to the 90-day history")

        plt.tight_layout()

        filename = (
            f"attention_{target.lower()}_"
            f"{p.replace(' ', '_')}.png"
        )

        fig.savefig(directory / filename, dpi=130)
        plt.close(fig)


def run_interpretation(art, fd, targets, directory):
    """Save TFT interpretation tables, driver summaries and figures."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    for target in targets:
        print(f"\nGenerating TFT interpretation for {target}...")

        out = interpret_tft(art, fd, target)

        for key in ("static", "encoder", "decoder"):
            out[key].round(4).to_csv(
                d / f"tft_{key}_weights_{target.lower()}.csv"
            )

        top_drivers(out).to_csv(
            d / f"tft_top_drivers_{target.lower()}.csv",
            index=False
        )

        plot_drivers(
            out,
            target,
            d / f"drivers_{target.lower()}.png"
        )

        plot_attention(
            out,
            fd,
            target,
            d
        )

    print(f"\nTFT interpretation outputs saved to: {d.resolve()}")

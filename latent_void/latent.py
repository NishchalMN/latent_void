import numpy as np


def latent_mask_from_gaussian_mask(gaussian_mask, latent_shape=None, downsample=8):
    mask = np.asarray(gaussian_mask).astype(bool)
    if mask.ndim == 1:
        if latent_shape is None:
            side = int(np.ceil(np.sqrt(mask.shape[0])))
            padded = np.zeros(side * side, dtype=bool)
            padded[:mask.shape[0]] = mask
            grid = padded.reshape(side, side)
        else:
            h, w = latent_shape[-2], latent_shape[-1]
            padded = np.zeros(h * w, dtype=bool)
            count = min(mask.shape[0], padded.shape[0])
            padded[:count] = mask[:count]
            grid = padded.reshape(h, w)
            return grid
    elif mask.ndim == 2:
        grid = mask
    else:
        raise ValueError("gaussian mask must be 1D or 2D")

    factor = int(downsample)
    if factor <= 1:
        return grid.astype(bool)
    h, w = grid.shape
    out_h = int(np.ceil(float(h) / factor))
    out_w = int(np.ceil(float(w) / factor))
    output = np.zeros((out_h, out_w), dtype=bool)
    for y in range(out_h):
        for x in range(out_w):
            patch = grid[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor]
            output[y, x] = bool(patch.any())
    return output


def expand_mask_to_latent(mask, latent):
    mask = np.asarray(mask).astype(bool)
    latent = np.asarray(latent)
    if mask.shape == latent.shape[-2:]:
        return mask
    h, w = latent.shape[-2], latent.shape[-1]
    y_scale = int(np.ceil(float(h) / mask.shape[0]))
    x_scale = int(np.ceil(float(w) / mask.shape[1]))
    expanded = np.kron(mask, np.ones((y_scale, x_scale), dtype=bool))
    return expanded[:h, :w]


def fallback_inpaint_latent(latent, mask):
    """Simple plumbing fallback: fill masked latent cells with unmasked means."""
    latent = np.array(latent, copy=True)
    mask = expand_mask_to_latent(mask, latent)
    if latent.ndim < 3:
        raise ValueError("latent must have shape [..., channels, height, width] or [channels, height, width]")
    channels_axis = -3
    moved = np.moveaxis(latent, channels_axis, 0)
    flat_mask = mask.reshape(-1)
    for channel_idx in range(moved.shape[0]):
        channel = moved[channel_idx]
        leading = int(np.prod(channel.shape[:-2])) if channel.ndim > 2 else 1
        channel_reshaped = channel.reshape((leading,) + channel.shape[-2:])
        for item_idx in range(leading):
            plane = channel_reshaped[item_idx]
            unmasked = plane[~mask]
            fill_value = float(unmasked.mean()) if unmasked.size else 0.0
            flat = plane.reshape(-1)
            flat[flat_mask] = fill_value
    return np.moveaxis(moved, 0, channels_axis)

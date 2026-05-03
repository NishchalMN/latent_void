import numpy as np


def _block_any_resize(mask, target_h, target_w):
    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape[-2:]
    output = np.zeros(mask.shape[:-2] + (int(target_h), int(target_w)), dtype=bool)
    for y in range(int(target_h)):
        y0 = int(np.floor(y * h / float(target_h)))
        y1 = int(np.ceil((y + 1) * h / float(target_h)))
        for x in range(int(target_w)):
            x0 = int(np.floor(x * w / float(target_w)))
            x1 = int(np.ceil((x + 1) * w / float(target_w)))
            output[..., y, x] = mask[..., y0:y1, x0:x1].any(axis=(-2, -1))
    return output


def _latent_mask_target_shape(latent_shape):
    if latent_shape is None:
        return None
    shape = tuple(int(value) for value in latent_shape)
    if len(shape) < 3:
        raise ValueError("latent_shape must include channel, height, and width dimensions")
    return shape[:-3] + shape[-2:]


def _legacy_downsample(grid, downsample):
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


def latent_mask_from_gaussian_mask(gaussian_mask, latent_shape=None, downsample=8, gaussian_grid_shape=None):
    mask = np.asarray(gaussian_mask).astype(bool)
    if gaussian_grid_shape is not None:
        grid_shape = tuple(int(value) for value in np.asarray(gaussian_grid_shape).tolist())
        if int(np.prod(grid_shape)) != mask.size:
            raise ValueError("gaussian_grid_shape product does not match gaussian mask length")
        grid = mask.reshape(grid_shape)
        if len(grid_shape) == 4:
            batch, views, height, width = grid_shape
            grid = grid.reshape(batch * views, height, width)
        elif len(grid_shape) == 3:
            grid = grid.reshape(grid_shape[0], grid_shape[1], grid_shape[2])
        elif len(grid_shape) != 2:
            raise ValueError("gaussian_grid_shape must be [B,V,H,W], [V,H,W], or [H,W]")

        target_shape = _latent_mask_target_shape(latent_shape)
        if target_shape is None:
            if grid.ndim == 3:
                return np.stack([_legacy_downsample(item, downsample) for item in grid], axis=0)
            return _legacy_downsample(grid, downsample)

        if grid.ndim == 2:
            if len(target_shape) != 2:
                raise ValueError("2D gaussian grid cannot map to latent leading dimensions %s" % (target_shape[:-2],))
            return _block_any_resize(grid, target_shape[-2], target_shape[-1])

        if tuple(grid.shape[:-2]) != tuple(target_shape[:-2]):
            raise ValueError("gaussian mask leading shape %s does not match latent mask leading shape %s" % (grid.shape[:-2], target_shape[:-2]))
        return _block_any_resize(grid, target_shape[-2], target_shape[-1])

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

    return _legacy_downsample(grid, downsample)


def expand_mask_to_latent(mask, latent):
    mask = np.asarray(mask).astype(bool)
    latent = np.asarray(latent)
    expected_shape = latent.shape[:-3] + latent.shape[-2:]
    if mask.shape == expected_shape:
        return mask
    if mask.shape == latent.shape[-2:]:
        if latent.ndim == 3:
            return mask
        return np.broadcast_to(mask, expected_shape).copy()
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
    for channel_idx in range(moved.shape[0]):
        channel = moved[channel_idx]
        unmasked = channel[~mask]
        fill_value = float(unmasked.mean()) if unmasked.size else 0.0
        channel[mask] = fill_value
    return np.moveaxis(moved, 0, channels_axis)


def _spatial_neighbor_average(array):
    padded = np.pad(array, [(0, 0)] * (array.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    return (
        padded[..., :-2, 1:-1]
        + padded[..., 2:, 1:-1]
        + padded[..., 1:-1, :-2]
        + padded[..., 1:-1, 2:]
    ) * 0.25


def context_inpaint_latent(latent, mask, iterations=128):
    """Harmonic latent fill that only updates masked cells from local context."""
    original = np.asarray(latent)
    output = np.array(original, copy=True)
    mask = expand_mask_to_latent(mask, output)
    if output.ndim < 3:
        raise ValueError("latent must have shape [..., channels, height, width] or [channels, height, width]")
    if not mask.any():
        return output

    moved = np.moveaxis(output, -3, 0)
    original_moved = np.moveaxis(original, -3, 0)
    iterations = max(1, int(iterations))
    for channel_idx in range(moved.shape[0]):
        channel = moved[channel_idx]
        original_channel = original_moved[channel_idx]
        unmasked = original_channel[~mask]
        fill_value = float(unmasked.mean()) if unmasked.size else 0.0
        channel[mask] = fill_value
        for _ in range(iterations):
            averaged = _spatial_neighbor_average(channel)
            channel[mask] = averaged[mask]
            channel[~mask] = original_channel[~mask]
    return np.moveaxis(moved, 0, -3)

import os
import glob
import argparse
import numpy as np
import cv2

def load_flow(path):
    flow = np.load(path)
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = np.transpose(flow, (1, 2, 0))
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Unexpected flow shape: {flow.shape}")
    return flow.astype(np.float32)

def flow_to_color(flow, clip_flow=None, epsilon=1e-6):
    """
    Legacy HSV flow visualization: hue=direction, saturation=255 fixed,
    value=magnitude normalized by this frame's own max. Kept exactly as
    it was (signature and behavior unchanged) for backward compatibility
    with any other caller -- see flow_to_color_bright() for the
    brighter/clearer alternative.
    """

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=False)

    if clip_flow is not None:
        mag = np.clip(mag, 0, clip_flow)

    mag_norm = mag / (mag.max() + epsilon)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = ((ang * 180 / np.pi) / 2).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = (mag_norm * 255).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


# Measured from outputs/town10hd_multiradar_density_validation frames
# 46-51 (Town10HD, ego moving ~30 km/h): magnitude p99 ~= 0.127-0.133,
# max ~= 0.151-0.158 across all 6 frames. CARLA's optical-flow sensor
# reports motion normalized to image size (not pixels), so this is on
# the order of 0.1-0.2, not 10-15 -- a fixed default must be measured
# per-dataset rather than guessed. Re-measure with flow_magnitude_stats()
# if used on a sequence with a very different ego speed / FPS.
DEFAULT_BRIGHT_MAX_FLOW = 0.15


def flow_magnitude_stats(flow, percentiles=(50, 90, 95, 99)):
    """
    min/mean/max plus the given percentiles of ||flow|| for one frame,
    used to pick a reproducible max_flow clip value instead of guessing.
    """

    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    pct_values = np.percentile(mag, percentiles)

    stats = {
        "mean": float(mag.mean()),
        "min": float(mag.min()),
        "max": float(mag.max()),
    }

    stats.update({f"p{p:g}": float(v) for p, v in zip(percentiles, pct_values)})

    return stats


def flow_to_color_bright(flow, max_flow=DEFAULT_BRIGHT_MAX_FLOW, percentile=None, epsilon=1e-6):
    """
    Brighter HSV flow visualization: hue=direction (unchanged from
    flow_to_color), saturation=clipped normalized magnitude,
    value=255 fixed.

    flow_to_color() maps magnitude -> value, so small/medium flow (most
    of a typical frame) ends up dark/near-black. Here magnitude instead
    controls saturation against a fixed brightness, so near-zero flow
    reads as pale/white rather than black, and motion structure (ego
    parallax, moving vs. co-moving objects) stays visible at any
    magnitude instead of being crushed toward black.

    max_flow: fixed clip value (flow/max_flow, clamped to [0,1]) so the
    same magnitude always maps to the same color across frames
    (reproducible). Pick this from flow_magnitude_stats() on
    representative frames, not an arbitrary guess.

    percentile: if given (e.g. 99), use this frame's own
    np.percentile(magnitude, percentile) as the clip value instead of a
    fixed max_flow -- a debug/exploration option; less reproducible
    frame-to-frame than a fixed max_flow since the clip value then
    depends on each frame's own content.
    """

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=False)

    if percentile is not None:
        clip_value = float(np.percentile(mag, percentile))
    else:
        clip_value = float(max_flow)

    clip_value = max(clip_value, epsilon)

    mag_norm = np.clip(mag / clip_value, 0.0, 1.0)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = ((ang * 180 / np.pi) / 2).astype(np.uint8)
    hsv[:, :, 1] = (mag_norm * 255).astype(np.uint8)
    hsv[:, :, 2] = 255

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def _make_middlebury_colorwheel():
    """
    The standard Middlebury/RAFT optical-flow color wheel: a 55-entry
    RGB lookup table built from 6 hue ramps (Red->Yellow->Green->Cyan->
    Blue->Magenta->Red). Same construction as the reference Middlebury
    `computeColor.m` / RAFT `flow_viz.make_colorwheel` -- NumPy only, no
    extra dependency.
    """

    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR

    wheel = np.zeros((ncols, 3), dtype=np.float64)
    col = 0

    wheel[0:RY, 0] = 255
    wheel[0:RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY

    wheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    wheel[col:col + YG, 1] = 255
    col += YG

    wheel[col:col + GC, 1] = 255
    wheel[col:col + GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC

    wheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    wheel[col:col + CB, 2] = 255
    col += CB

    wheel[col:col + BM, 2] = 255
    wheel[col:col + BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col += BM

    wheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    wheel[col:col + MR, 0] = 255

    return wheel


_MIDDLEBURY_COLORWHEEL = _make_middlebury_colorwheel()


def flow_to_color_middlebury(flow, max_flow=DEFAULT_BRIGHT_MAX_FLOW, percentile=None, epsilon=1e-6):
    """
    Middlebury/RAFT-style flow color wheel: direction -> hue via the
    standard 55-color wheel, magnitude -> saturation/intensity toward
    that hue, zero flow -> white. Same fixed-scale philosophy as
    flow_to_color_bright() (NOT frame-local-max normalized): magnitude
    is clipped against `max_flow` so the same physical flow always maps
    to the same color across frames.

    max_flow / percentile: same meaning as in flow_to_color_bright().
    """

    u = flow[:, :, 0].astype(np.float64)
    v = flow[:, :, 1].astype(np.float64)

    mag = np.sqrt(u ** 2 + v ** 2)

    if percentile is not None:
        clip_value = float(np.percentile(mag, percentile))
    else:
        clip_value = float(max_flow)

    clip_value = max(clip_value, epsilon)

    # Radius (saturation driver): fixed scaling, not frame-local max.
    rad = np.clip(mag / clip_value, 0.0, 1.0)

    # Middlebury convention (matches the reference implementation
    # exactly, including the sign flip -- this is what makes the wheel
    # colors match the canonical Middlebury/RAFT figures).
    angle = np.arctan2(-v, -u) / np.pi  # range (-1, 1]

    ncols = _MIDDLEBURY_COLORWHEEL.shape[0]
    fk = (angle + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    rgb = np.zeros((*u.shape, 3), dtype=np.uint8)

    for channel in range(3):
        col0 = _MIDDLEBURY_COLORWHEEL[k0, channel] / 255.0
        col1 = _MIDDLEBURY_COLORWHEEL[k1, channel] / 255.0
        col = (1.0 - f) * col0 + f * col1

        in_range = rad <= 1.0
        # rad=0 -> col_out=1 (white); rad=1 -> col_out=col (full hue).
        col_out = np.where(in_range, 1.0 - rad * (1.0 - col), col * 0.75)

        rgb[:, :, channel] = np.floor(255.0 * np.clip(col_out, 0.0, 1.0)).astype(np.uint8)

    bgr = rgb[:, :, ::-1].copy()
    return bgr


def draw_flow_legend(image, size=56, margin=10):
    """
    Small corner color-wheel legend (hue -> direction) so the flow panel
    is self-explanatory. Drawn in-place into the top-right corner;
    returns the same image. Deliberately tiny so it doesn't obscure the
    main flow field.
    """

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center = (size - 1) / 2.0

    dx = xx - center
    dy = yy - center

    angle = np.arctan2(dy, dx)
    angle[angle < 0] += 2 * np.pi

    radius = np.sqrt(dx ** 2 + dy ** 2)
    inside = radius <= center

    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[:, :, 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
    hsv[:, :, 1] = np.clip(radius / center * 255, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = 255

    wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    wheel[~inside] = (25, 25, 25)

    cv2.circle(wheel, (int(center), int(center)), int(center), (255, 255, 255), 1, cv2.LINE_AA)

    height, width = image.shape[:2]
    y0 = margin
    x0 = width - size - margin - 14

    if x0 < 14 or y0 + size >= height:
        return image  # panel too small for the legend; skip rather than crash/overflow

    image[y0:y0 + size, x0:x0 + size] = wheel

    cx_img, cy_img = x0 + int(center), y0 + int(center)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image, "R", (x0 + size + 2, cy_img + 4), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "L", (max(x0 - 12, 0), cy_img + 4), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "D", (cx_img - 4, y0 + size + 12), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "U", (cx_img - 4, max(y0 - 4, 10)), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return image

def _render(flow, mode, clip_flow, max_flow, percentile, legend):
    if mode == "bright":
        color = flow_to_color_bright(flow, max_flow=max_flow, percentile=percentile)
    else:
        color = flow_to_color(flow, clip_flow=clip_flow)

    if legend:
        color = draw_flow_legend(color)

    return color


def visualize_one(npy_path, save_path=None, clip_flow=None, show=True, mode="legacy", max_flow=DEFAULT_BRIGHT_MAX_FLOW, percentile=None, legend=False):
    flow = load_flow(npy_path)
    color = _render(flow, mode, clip_flow, max_flow, percentile, legend)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, color)

    if show:
        cv2.imshow("Optical Flow", color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def visualize_directory(input_dir, output_dir=None, clip_flow=None, max_files=None, mode="legacy", max_flow=DEFAULT_BRIGHT_MAX_FLOW, percentile=None, legend=False):
    paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if max_files is not None:
        paths = paths[:max_files]

    if len(paths) == 0:
        raise FileNotFoundError(f"No npy files found in: {input_dir}")

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0] + ".png"
        save_path = None if output_dir is None else os.path.join(output_dir, name)

        flow = load_flow(path)
        color = _render(flow, mode, clip_flow, max_flow, percentile, legend)

        if save_path is not None:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(save_path, color)

        print(f"saved: {save_path if save_path is not None else name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--clip_flow", type=float, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    # New, opt-in: the default ("legacy") reproduces the original
    # flow_to_color() output exactly, so existing usage is unaffected.
    parser.add_argument("--mode", type=str, default="legacy", choices=["legacy", "bright"], help="legacy = original magnitude->value mapping; bright = new direction-hue/magnitude-saturation mapping")
    parser.add_argument("--max-flow", type=float, default=DEFAULT_BRIGHT_MAX_FLOW, help="Fixed magnitude clip for --mode bright (see flow_magnitude_stats to measure your own sequence's scale)")
    parser.add_argument("--flow-percentile", type=float, default=None, help="Use this frame's own magnitude percentile as the --mode bright clip instead of --max-flow (debug option)")
    parser.add_argument("--legend", action="store_true", help="Draw a small direction color-wheel legend in the corner")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        save_path = args.output
        if save_path is None:
            base = os.path.splitext(args.input)[0] + "_vis.png"
            save_path = base
        visualize_one(args.input, save_path=save_path, clip_flow=args.clip_flow, show=args.show, mode=args.mode, max_flow=args.max_flow, percentile=args.flow_percentile, legend=args.legend)
        print(f"saved: {save_path}")
    else:
        visualize_directory(args.input, output_dir=args.output, clip_flow=args.clip_flow, max_files=args.max_files, mode=args.mode, max_flow=args.max_flow, percentile=args.flow_percentile, legend=args.legend)

if __name__ == "__main__":
    main()
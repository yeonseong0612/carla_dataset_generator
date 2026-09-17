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

def visualize_one(npy_path, save_path=None, clip_flow=None, show=True):
    flow = load_flow(npy_path)
    color = flow_to_color(flow, clip_flow=clip_flow)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, color)

    if show:
        cv2.imshow("Optical Flow", color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def visualize_directory(input_dir, output_dir=None, clip_flow=None, max_files=None):
    paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if max_files is not None:
        paths = paths[:max_files]

    if len(paths) == 0:
        raise FileNotFoundError(f"No npy files found in: {input_dir}")

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0] + ".png"
        save_path = None if output_dir is None else os.path.join(output_dir, name)

        flow = load_flow(path)
        color = flow_to_color(flow, clip_flow=clip_flow)

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
    args = parser.parse_args()

    if os.path.isfile(args.input):
        save_path = args.output
        if save_path is None:
            base = os.path.splitext(args.input)[0] + "_vis.png"
            save_path = base
        visualize_one(args.input, save_path=save_path, clip_flow=args.clip_flow, show=args.show)
        print(f"saved: {save_path}")
    else:
        visualize_directory(args.input, output_dir=args.output, clip_flow=args.clip_flow, max_files=args.max_files)

if __name__ == "__main__":
    main()
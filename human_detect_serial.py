#!/usr/bin/env python3
"""
human_detect_serial.py
──────────────────────
Runs YOLO pose detection on the camera feed and sends a single byte over
serial each frame:
    b'1'  → at least one human detected
    b'0'  → no human detected

Designed to run on the Jetson.  Uses CUDA automatically if available.

Usage:
    python3 human_detect_serial.py
    python3 human_detect_serial.py --no-serial     # test without hardware
    python3 human_detect_serial.py --port /dev/ttyUSB0 --baud 115200
    python3 human_detect_serial.py --camera 1      # use a different camera index
"""

import argparse
import sys
import time

import cv2
from ultralytics import YOLO

# ─────────────────────────────────────────── CONFIG (override via CLI args)
DEFAULT_PORT   = "/dev/ttyTHS0"   # Jetson UART; use /dev/ttyUSB0 for USB adapter
DEFAULT_BAUD   = 9600
DEFAULT_CAMERA = 0
DEFAULT_MODEL  = "yolo11n-pose.pt"   # smallest + fastest; swap for yolo11s-pose.pt etc.
CONF_THRESHOLD = 0.40                # detection confidence threshold


def main():
    parser = argparse.ArgumentParser(description="Human detection → serial flag")
    parser.add_argument("--port",      default=DEFAULT_PORT,   help="Serial port")
    parser.add_argument("--baud",      default=DEFAULT_BAUD,   type=int)
    parser.add_argument("--camera",    default=DEFAULT_CAMERA, type=int)
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    parser.add_argument("--conf",      default=CONF_THRESHOLD, type=float)
    parser.add_argument("--no-serial", action="store_true",
                        help="Disable serial output (useful for testing)")
    args = parser.parse_args()

    # ── Serial setup ─────────────────────────────────────────────────────
    ser = None
    if not args.no_serial:
        try:
            import serial
            ser = serial.Serial(args.port, args.baud, timeout=0)
            print(f"Serial open: {args.port} @ {args.baud} baud")
        except ImportError:
            sys.exit("pyserial not installed. Run: pip install pyserial")
        except Exception as e:
            sys.exit(f"Cannot open serial port {args.port}: {e}")
    else:
        print("Serial disabled (--no-serial mode)")

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"Loading {args.model} ...")
    model = YOLO(args.model)
    # YOLO auto-selects CUDA on Jetson; force CPU with device='cpu' if needed
    print("Model ready.\n")

    # ── Camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}")
    print(f"Camera {args.camera} open. Press q or ESC to quit.\n")

    prev_flag = None   # track last sent value to avoid spamming identical bytes
    t0, frames = time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — retrying...")
            time.sleep(0.05)
            continue

        # ── Inference ─────────────────────────────────────────────────────
        results = model.predict(frame, conf=args.conf, verbose=False, classes=[0])
        r = results[0] if results else None

        human_present = (
            r is not None
            and r.boxes is not None
            and len(r.boxes) > 0
        )
        flag = 1 if human_present else 0

        # ── Serial output ─────────────────────────────────────────────────
        # Send on every change (edge) so the receiver doesn't need to parse
        # a continuous stream.  Remove the `flag != prev_flag` check if the
        # receiver expects a heartbeat every frame instead.
        if flag != prev_flag:
            if ser is not None:
                ser.write(bytes([flag]))
            prev_flag = flag
            print(f"→ {flag}  ({'HUMAN' if flag else 'none':5s})")

        # ── Display (optional — comment out if running headless) ──────────
        annotated = r.plot() if r is not None else frame
        frames += 1
        fps = frames / max(0.001, time.time() - t0)
        cv2.putText(annotated, f"{fps:.1f} fps  {'HUMAN' if flag else '---'}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if flag else (0, 80, 80), 2)
        cv2.imshow("human_detect", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()


if __name__ == "__main__":
    main()

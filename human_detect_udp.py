#!/usr/bin/env python3
"""
human_detect_udp.py
───────────────────
Runs YOLO pose detection on the Jetson camera and broadcasts a UDP packet
each time the human-present state changes:
    b'1'  → at least one human detected
    b'0'  → no human detected
The receiver (any device on the same network) just opens a UDP socket and
reads one byte — no libraries, no connection setup needed on that side.
Usage:
    python3 human_detect_udp.py --receiver 192.168.1.50
    python3 human_detect_udp.py --receiver 192.168.1.50 --port 5005
    python3 human_detect_udp.py --receiver 192.168.1.50 --every-frame  # heartbeat mode
    python3 human_detect_udp.py --receiver 255.255.255.255             # LAN broadcast
"""

import argparse
import socket
import sys
import time

import cv2
from ultralytics import YOLO

# ──────────────────────────────────────────────────── CONFIG
DEFAULT_RECEIVER = "192.168.50.2"   # LAN broadcast; replace with specific IP
DEFAULT_UDP_PORT = 8221
DEFAULT_CAMERA   = 0
DEFAULT_MODEL    = "yolo11n-pose.pt"
CONF_THRESHOLD   = 0.40


def main():
    parser = argparse.ArgumentParser(description="Human detection → UDP flag")
    parser.add_argument("--receiver",    default=DEFAULT_RECEIVER,
                        help="Receiver IP (or 255.255.255.255 for LAN broadcast)")
    parser.add_argument("--port",        default=DEFAULT_UDP_PORT, type=int,
                        help="UDP destination port")
    parser.add_argument("--camera",      default=DEFAULT_CAMERA,   type=int)
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--conf",        default=CONF_THRESHOLD,   type=float)
    parser.add_argument("--every-frame", action="store_true",
                        help="Send a packet every frame instead of only on change")
    args = parser.parse_args()

    # ── UDP socket ────────────────────────────────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Allow broadcast if using 255.255.255.255
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"UDP → {args.receiver}:{args.port}")

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"Loading {args.model} ...")
    model = YOLO(args.model)   # CUDA auto-selected on Jetson
    print("Model ready.\n")

    # ── Camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}")
    print(f"Camera {args.camera} open. Press q or ESC to quit.\n")

    prev_flag = None
    t0, frames = time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ── Inference ─────────────────────────────────────────────────────
        results = model.predict(frame, conf=args.conf, verbose=False, classes=[0])
        r = results[0] if results else None

        human_present = (r is not None
                         and r.boxes is not None
                         and len(r.boxes) > 0)
        flag = 1 if human_present else 0

        # ── UDP send ──────────────────────────────────────────────────────
        if args.every_frame or flag != prev_flag:
            sock.sendto(bytes([flag]), (args.receiver, args.port))
            prev_flag = flag
            print(f"→ {flag}  ({'HUMAN' if flag else 'none':5s})")

        # ── Display (comment out for headless) ────────────────────────────
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
    sock.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
human_detect_receiver.py
────────────────────────
Drop this into your app and instantiate HumanDetectReceiver.
The UDP listener runs in a daemon thread — your main loop is never blocked.

Usage in your app:

    from human_detect_receiver import HumanDetectReceiver

    detector = HumanDetectReceiver(port=8221)
    detector.start()

    while True:
        if detector.human_present:
            print("do something")
        time.sleep(0.1)
"""

import socket
import threading
import time


class HumanDetectReceiver:
    def __init__(self, port: int = 8221, timeout: float = 2.0):
        """
        port    — must match DEFAULT_UDP_PORT on the Jetson sender
        timeout — seconds after the last '1' packet before human_present
                  resets to False automatically (guards against packet loss)
        """
        self._port           = port
        self._timeout        = timeout
        self._human_present  = False
        self._last_seen      = 0.0
        self._lock           = threading.Lock()
        self._thread         = threading.Thread(target=self._listen, daemon=True)

    # ── public API ────────────────────────────────────────────────────────

    def start(self):
        """Start the background listener thread."""
        self._thread.start()

    @property
    def human_present(self) -> bool:
        """True if a human was detected within the last `timeout` seconds."""
        with self._lock:
            if self._human_present and (time.time() - self._last_seen > self._timeout):
                self._human_present = False   # auto-clear on packet loss / timeout
            return self._human_present

    # ── internal ──────────────────────────────────────────────────────────

    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", self._port))
        sock.settimeout(0.5)   # unblocks periodically so the thread can exit cleanly
        print(f"[HumanDetectReceiver] listening on UDP port {self._port}")

        while True:
            try:
                data, _ = sock.recvfrom(16)
                with self._lock:
                    if data[0] == 1:
                        self._human_present = True
                        self._last_seen = time.time()
                    else:
                        self._human_present = False
            except socket.timeout:
                pass   # just loop again

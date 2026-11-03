"""Deterministic synthetic traffic generator for tests and demos."""

from __future__ import annotations

import random
from typing import Iterator

from .base import Packet, PacketSource


class SyntheticSource(PacketSource):
    """Generates benign web traffic with an optional embedded port scan."""

    def __init__(self, n_packets: int = 500, seed: int = 0, inject_scan: bool = True):
        self.n_packets = n_packets
        self.seed = seed
        self.inject_scan = inject_scan

    def __iter__(self) -> Iterator[Packet]:
        rng = random.Random(self.seed)
        ts = 0.0
        for _ in range(self.n_packets):
            ts += rng.expovariate(50.0)
            yield Packet(
                ts=ts,
                src_ip=f"10.0.0.{rng.randint(2, 20)}",
                dst_ip="10.0.0.1",
                src_port=rng.randint(32768, 60999),
                dst_port=rng.choice([80, 443, 53]),
                proto="tcp",
                length=rng.randint(60, 1500),
                flags="PA",
            )
        if self.inject_scan:
            for port in range(1, 201):
                ts += 0.001
                yield Packet(
                    ts=ts,
                    src_ip="192.168.1.66",
                    dst_ip="10.0.0.1",
                    src_port=40000,
                    dst_port=port,
                    proto="tcp",
                    length=60,
                    flags="S",
                )

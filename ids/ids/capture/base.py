"""Packet abstraction and source interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class Packet:
    """Minimal protocol-agnostic packet view used by the rest of the pipeline."""

    ts: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: str  # "tcp" | "udp" | "icmp" | "other"
    length: int
    flags: str = ""  # TCP flags as a string such as "S", "SA", "F"
    payload: bytes = field(default=b"", repr=False)

    @property
    def flow_key(self) -> tuple:
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.proto)


class PacketSource:
    """Iterable producing Packet objects. Subclasses implement __iter__."""

    def __iter__(self) -> Iterator[Packet]:  # pragma: no cover - interface
        raise NotImplementedError

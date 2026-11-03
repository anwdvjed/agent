"""PCAP file source. Requires the optional `scapy` dependency."""

from __future__ import annotations

from typing import Iterator

from .base import Packet, PacketSource


class PcapSource(PacketSource):
    def __init__(self, path: str):
        self.path = path

    def __iter__(self) -> Iterator[Packet]:
        try:
            from scapy.all import IP, TCP, UDP, PcapReader  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PcapSource requires `pip install ids[live]`") from exc

        with PcapReader(self.path) as reader:
            for pkt in reader:
                if IP not in pkt:
                    continue
                ip = pkt[IP]
                proto, sport, dport, flags = "other", 0, 0, ""
                if TCP in pkt:
                    proto, sport, dport, flags = "tcp", pkt[TCP].sport, pkt[TCP].dport, str(pkt[TCP].flags)
                elif UDP in pkt:
                    proto, sport, dport = "udp", pkt[UDP].sport, pkt[UDP].dport
                yield Packet(
                    ts=float(pkt.time),
                    src_ip=ip.src,
                    dst_ip=ip.dst,
                    src_port=sport,
                    dst_port=dport,
                    proto=proto,
                    length=len(pkt),
                    flags=flags,
                    payload=bytes(pkt[IP].payload),
                )

from scapy.all import rdpcap, Packet as ScapyPacket
from scapy.layers.inet import IP, UDP, TCP
from typing import Any, List, Tuple

# Let's start with just parsing packets and not bothering with defining flows or a state machine or anything like that. We'll start with TCP and UDP.
# Also, let's do a better job with type definitions and so forth.
#

def parse_packet():
    pass

def parse_tcp():
    pass

def parse_udp(packet: ScapyPacket):
    udp_layer = parse_udp_header(packet=packet)
    if udp_layer is not None:
        sport, dport, u_len, chksum = udp_layer
    else:
        # handle error here
        pass

def parse_ip_layer():
    pass

def parse_udp_header(packet: ScapyPacket) -> tuple[int, int, int | None, int | None] | None:
    try:
        if packet.haslayer(UDP):
            udp_layer: UDP = packet[UDP] # pyright: ignore[reportAny]
            sport: int = udp_layer.sport  # pyright: ignore[reportAny]
            dport: int = udp_layer.dport  # pyright: ignore[reportAny]
            u_len: int | None = udp_layer.len  # pyright: ignore[reportAny]
            chksum: int | None = udp_layer.chksum  # pyright: ignore[reportAny]
            return sport, dport, u_len, chksum
    except AttributeError as error:
        print(error)

def parse_pcap():
    pass

# for now, let's use the bpf filter to just focus on UDP packets
def read_pcap(input_file: str, bpf_filter: str = "udp"):
    packets = rdpcap(input_file)
    for packet in packets:
        if bpf_filter == "udp" and packet.haslayer(UDP):
            parse_udp(packet=packet)
        elif bpf_filter != "udp":
            # Handle other filters or no filter
            parse_udp(packet=packet)

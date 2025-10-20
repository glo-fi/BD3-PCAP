from box import Box
from scapy.all import Packet as ScapyPacket
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.inet6 import IPv6


class ProcessedPacket:
    """
    Once a ScapyPacket is processed, it is converted into a ProcessedPacket.
    Continas both features and helper functions.
    """

    def __init__(self, packet: ScapyPacket):
        self.features = Box(
            {
                # Universal packet features
                "packet_size": None,
                "timestamp": None,
                # IP layer features (most packets have IP)
                "ip_version": None,
                "ip_ihl": None,
                "ip_tos": None,
                "ip_len": None,
                "ip_id": None,
                "ip_flags": None,
                "ip_frag": None,
                "ip_ttl": None,
                "ip_proto": None,
                "ip_chksum": None,
                "src_ip": None,
                "dst_ip": None,
                # IPv6-specific features
                "ipv6_version": None,
                "ipv6_tc": None,  # Traffic class
                "ipv6_fl": None,  # Flow label
                "ipv6_plen": None,  # Payload length
                "ipv6_nh": None,  # Next header
                "ipv6_hlim": None,  # Hop limit
                # Transport layer features (TCP/UDP)
                "src_port": None,
                "dst_port": None,
                "protocol": None,
                # TCP-specific features
                "tcp_seq": None,
                "tcp_ack": None,
                "tcp_dataofs": None,
                "tcp_reserved": None,
                "tcp_flags": None,
                "tcp_window": None,
                "tcp_chksum": None,
                "tcp_urgptr": None,
                # UDP-specific features
                "udp_len": None,
                "udp_chksum": None,
                # Payload features
                "payload_size": 0,
                "has_payload": False,
            }
        )
        self.extract_packet_features(packet)
        self.flow_id = self.get_flow_id()

    def get_flow_id(self) -> str:
        """
        Returns a unique flow id
        Normalizes flow direction by sorting IPs to ensure bidirectional flows
        have the same ID regardless of packet direction.
        """
        if bool(self.features.ip_version):
            # For IPv6, wrap addresses in brackets for proper port notation
            if self.features.ip_version == 6:
                src_addr = f"[{self.features.src_ip}]"
                dst_addr = f"[{self.features.dst_ip}]"
            else:
                src_addr = self.features.src_ip
                dst_addr = self.features.dst_ip

            # Create tuple for comparison to normalize flow direction
            src_tuple = (self.features.src_ip, self.features.src_port)
            dst_tuple = (self.features.dst_ip, self.features.dst_port)

            if src_tuple < dst_tuple:
                return f"{src_addr}:{self.features.src_port}->{dst_addr}:{self.features.dst_port}_{self.features.protocol}"
            else:
                return f"{dst_addr}:{self.features.dst_port}->{src_addr}:{self.features.src_port}_{self.features.protocol}"
        else:
            print("[!] Not IP packet!")
            return ""

    def extract_packet_features(self, packet: ScapyPacket):
        """Extract standardized ML features from any packet type"""

        self.features.packet_size = len(packet)
        self.features.timestamp = float(packet.time) if hasattr(packet, "time") else 0.0

        # Extract IP layer features
        if packet.haslayer(IP):
            self.extract_ip_features(packet)
        elif packet.haslayer(IPv6):
            self.extract_ipv6_features(packet)

        # Extract transport layer features
        if packet.haslayer(TCP):
            self.extract_tcp_features(packet)
            self.features.protocol = "TCP"
        elif packet.haslayer(UDP):
            self.extract_udp_features(packet)
            self.features.protocol = "UDP"

    def extract_tcp_features(self, packet: ScapyPacket):
        """Extract TCP-specific features for ML"""

        if packet.haslayer(TCP):
            tcp_layer: TCP = packet[TCP]
            self.features.src_port = tcp_layer.sport
            self.features.dst_port = tcp_layer.dport
            self.features.tcp_seq = tcp_layer.seq
            self.features.tcp_ack = tcp_layer.ack
            self.features.tcp_dataofs = tcp_layer.dataofs
            self.features.tcp_reserved = tcp_layer.reserved
            self.features.tcp_flags = tcp_layer.flags
            self.features.tcp_window = tcp_layer.window
            self.features.tcp_chksum = tcp_layer.chksum
            self.features.tcp_urgptr = tcp_layer.urgptr
            self.features.payload_size = (
                len(tcp_layer.payload) if tcp_layer.payload else 0
            )
            self.features.has_payload = bool(tcp_layer.payload)

    def extract_udp_features(self, packet: ScapyPacket):
        """Extract UDP-specific features for ML"""

        if packet.haslayer(UDP):
            udp_layer: UDP = packet[UDP]
            self.features.src_port = udp_layer.sport
            self.features.dst_port = udp_layer.dport
            self.features.udp_len = udp_layer.len
            self.features.udp_chksum = udp_layer.chksum
            self.features.payload_size = (
                len(udp_layer.payload) if udp_layer.payload else 0
            )
            self.features.has_payload = bool(udp_layer.payload)

    def extract_ip_features(self, packet: ScapyPacket):
        """Extract IP layer features for ML"""

        if packet.haslayer(IP):
            ip_layer: IP = packet[IP]
            self.features.ip_version = ip_layer.version
            self.features.ip_ihl = ip_layer.ihl
            self.features.ip_tos = ip_layer.tos
            self.features.ip_len = ip_layer.len
            self.features.ip_id = ip_layer.id
            self.features.ip_flags = ip_layer.flags
            self.features.ip_frag = ip_layer.frag
            self.features.ip_ttl = ip_layer.ttl
            self.features.ip_proto = ip_layer.proto
            self.features.ip_chksum = ip_layer.chksum
            self.features.src_ip = str(ip_layer.src)
            self.features.dst_ip = str(ip_layer.dst)

    def extract_ipv6_features(self, packet: ScapyPacket):
        """Extract IPv6 layer features for ML"""

        if packet.haslayer(IPv6):
            ipv6_layer: IPv6 = packet[IPv6]
            self.features.ipv6_version = ipv6_layer.version
            self.features.ipv6_tc = ipv6_layer.tc
            self.features.ipv6_fl = ipv6_layer.fl
            self.features.ipv6_plen = ipv6_layer.plen
            self.features.ipv6_nh = ipv6_layer.nh
            self.features.ipv6_hlim = ipv6_layer.hlim
            self.features.src_ip = str(ipv6_layer.src)
            self.features.dst_ip = str(ipv6_layer.dst)
            # Set ip_version for compatibility with flow_id generation
            self.features.ip_version = 6

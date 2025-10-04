from scapy.all import Packet as ScapyPacket, sniff
from scapy.layers.inet import IP, UDP, TCP
from typing import Any, List, Tuple, Dict, Optional
import csv

# Let's start with just parsing packets and not bothering with defining flows or a state machine or anything like that. We'll start with TCP and UDP.
# Also, let's do a better job with type definitions and so forth.
#


def extract_packet_features(packet: ScapyPacket) -> Dict[str, Any]:
    """Extract standardized ML features from any packet type"""
    features = {
        # Universal packet features
        "packet_size": len(packet),
        "timestamp": float(packet.time) if hasattr(packet, "time") else 0.0,
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

    # Extract IP layer features
    if packet.haslayer(IP):
        ip_features = extract_ip_features(packet)
        features.update(ip_features)

    # Extract transport layer features
    if packet.haslayer(TCP):
        tcp_features = extract_tcp_features(packet)
        features.update(tcp_features)
        features["protocol"] = "TCP"
    elif packet.haslayer(UDP):
        udp_features = extract_udp_features(packet)
        features.update(udp_features)
        features["protocol"] = "UDP"

    return features


def extract_tcp_features(packet: ScapyPacket) -> Dict[str, Any]:
    """Extract TCP-specific features for ML"""
    features = {}

    if packet.haslayer(TCP):
        tcp_layer: TCP = packet[TCP]
        features.update(
            {
                "src_port": tcp_layer.sport,
                "dst_port": tcp_layer.dport,
                "tcp_seq": tcp_layer.seq,
                "tcp_ack": tcp_layer.ack,
                "tcp_dataofs": tcp_layer.dataofs,
                "tcp_reserved": tcp_layer.reserved,
                "tcp_flags": tcp_layer.flags,
                "tcp_window": tcp_layer.window,
                "tcp_chksum": tcp_layer.chksum,
                "tcp_urgptr": tcp_layer.urgptr,
                "payload_size": len(tcp_layer.payload) if tcp_layer.payload else 0,
                "has_payload": bool(tcp_layer.payload),
            }
        )

    return features


def extract_udp_features(packet: ScapyPacket) -> Dict[str, Any]:
    """Extract UDP-specific features for ML"""
    features = {}

    if packet.haslayer(UDP):
        udp_layer: UDP = packet[UDP]
        features.update(
            {
                "src_port": udp_layer.sport,
                "dst_port": udp_layer.dport,
                "udp_len": udp_layer.len,
                "udp_chksum": udp_layer.chksum,
                "payload_size": len(udp_layer.payload) if udp_layer.payload else 0,
                "has_payload": bool(udp_layer.payload),
            }
        )

    return features


def extract_ip_features(packet: ScapyPacket) -> Dict[str, Any]:
    """Extract IP layer features for ML"""
    features = {}

    if packet.haslayer(IP):
        ip_layer: IP = packet[IP]
        features.update(
            {
                "ip_version": ip_layer.version,
                "ip_ihl": ip_layer.ihl,
                "ip_tos": ip_layer.tos,
                "ip_len": ip_layer.len,
                "ip_id": ip_layer.id,
                "ip_flags": ip_layer.flags,
                "ip_frag": ip_layer.frag,
                "ip_ttl": ip_layer.ttl,
                "ip_proto": ip_layer.proto,
                "ip_chksum": ip_layer.chksum,
                "src_ip": str(ip_layer.src),
                "dst_ip": str(ip_layer.dst),
            }
        )

    return features


def process_pcap(
    input_file: str,
    bpf_filter: str = "",
    output_file: Optional[str] = None,
    flush_interval: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Process pcap file and return ML-ready feature vectors

    Args:
        input_file: Path to pcap file
        bpf_filter: BPF filter string for packet filtering
        output_file: Optional path to CSV output file for periodic flushing
        flush_interval: Number of packets after which to flush to output_file
    """
    packets = sniff(offline=input_file, filter=bpf_filter)
    feature_vectors = []
    packet_count = 0
    csv_writer = None
    csv_file = None

    # Initialize CSV file if output is requested
    if output_file:
        csv_file = open(output_file, "w", newline="")
        # We'll write the header after processing the first packet to get field names

    for packet in packets:
        features = extract_packet_features(packet)
        feature_vectors.append(features)
        packet_count += 1

        # Initialize CSV writer with header after first packet
        if output_file and csv_file is not None and csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=features.keys())
            csv_writer.writeheader()

        # Flush to file every N packets if configured
        if (
            flush_interval
            and csv_file is not None
            and csv_writer
            and packet_count % flush_interval == 0
        ):
            csv_writer.writerows(feature_vectors)
            csv_file.flush()
            feature_vectors = []  # Clear the buffer after flushing

    # Flush any remaining features
    if csv_writer and csv_file is not None and feature_vectors:
        csv_writer.writerows(feature_vectors)
        csv_file.flush()

    # Close CSV file
    if csv_file:
        csv_file.close()

    return feature_vectors

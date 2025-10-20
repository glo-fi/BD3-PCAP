from scapy.all import sniff
from flow_diffusion.ppacket import ProcessedPacket
from typing import Any, List, Dict, Optional
import csv

# Let's start with just parsing packets and not bothering with defining flows or a state machine or anything like that. We'll start with TCP and UDP.
# Also, let's do a better job with type definitions and so forth.
#


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
        ppacket = ProcessedPacket(packet)
        features = ppacket.features
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

#!/usr/bin/env python3
"""SUPERSEDED - kept for reference only.

This was the original prototype: shell out to tshark, summarise one pcap
over a fixed window, append a row to a CSV. It is no longer part of the
pipeline and will not work against current simulation output, which no
longer writes pcap at all - the simulator records observations directly
from the PHY into observation.csv (smaller, exact signal/noise values, no
tshark dependency).

Use scripts/build_dataset.py instead.
"""

import argparse
import csv
import subprocess
from pathlib import Path


def run_tshark(pcap: Path, start: float, end: float):
    display_filter = f"frame.time_relative >= {start} && frame.time_relative < {end}"

    cmd = [
        "tshark",
        "-r", str(pcap),
        "-Y", display_filter,
        "-T", "fields",
        "-e", "frame.time_relative",
        "-e", "frame.len",
        "-e", "wlan.fc.type",
        "-e", "wlan.fc.type_subtype",
    ]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        while len(parts) < 4:
            parts.append("")

        time_s, frame_len, wlan_type, wlan_subtype = parts[:4]

        rows.append({
            "time": float(time_s) if time_s else None,
            "len": int(frame_len) if frame_len else 0,
            "type": wlan_type,
            "subtype": wlan_subtype,
        })

    return rows


def extract_features(pcap: Path, start: float, end: float):
    rows = run_tshark(pcap, start, end)

    packet_count = len(rows)
    byte_count = sum(r["len"] for r in rows)

    management_count = sum(1 for r in rows if r["type"] == "0")
    control_count = sum(1 for r in rows if r["type"] == "1")
    data_count = sum(1 for r in rows if r["type"] == "2")

    beacon_count = sum(1 for r in rows if r["subtype"] == "8")

    mean_frame_len = byte_count / packet_count if packet_count else 0

    return {
        "pcap": str(pcap),
        "window_start": start,
        "window_end": end,
        "packet_count": packet_count,
        "byte_count": byte_count,
        "mean_frame_len": mean_frame_len,
        "management_count": management_count,
        "control_count": control_count,
        "data_count": data_count,
        "beacon_count": beacon_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap", type=Path)
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--end", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("features.csv"))
    args = parser.parse_args()

    features = extract_features(args.pcap, args.start, args.end)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    write_header = not args.out.exists()
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(features.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(features)

    print(features)


if __name__ == "__main__":
    main()
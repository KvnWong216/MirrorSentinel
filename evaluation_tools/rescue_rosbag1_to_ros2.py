#!/usr/bin/env python3
"""Rescue a damaged-index rosbag1 file into a ROS2 rosbag.

This is a ROS2 Humble friendly path for old `.bag` files whose payload chunks
are readable but whose tail index is broken. It does not import ROS1 `rosbag`.

The script sequentially scans rosbag1 records, deserializes ROS1 wire data with
`rosbags`, serializes CDR, and writes rosbag2 sqlite3/mcap output.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from rosbags.interfaces import MessageDefinition, MessageDefinitionFormat
from rosbags.interfaces import (
    Qos,
    QosDurability,
    QosHistory,
    QosLiveliness,
    QosReliability,
    QosTime,
)
from rosbags.rosbag1.reader import (
    Header,
    ReaderError,
    RecordType,
    decompressors,
    read_bytes,
    read_uint32,
)
from rosbags.rosbag2 import StoragePlugin
from rosbags.rosbag2 import Writer as Rosbag2Writer
from rosbags.serde import SerdeError
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from rosbags.typesys.msg import normalize_msgtype


STATIC_MSGTYPE_RENAMES = {
    "tf/msg/tfMessage": "tf2_msgs/msg/TFMessage",
}

DEFAULT_QOS = [
    Qos(
        QosHistory.KEEP_LAST,
        10,
        QosReliability.RELIABLE,
        QosDurability.VOLATILE,
        QosTime(0, 0),
        QosTime(0, 0),
        QosLiveliness.AUTOMATIC,
        QosTime(0, 0),
        avoid_ros_namespace_conventions=False,
    )
]


@dataclass
class SourceConnection:
    conn_id: int
    topic: str
    msgtype: str
    msgdef: str
    md5sum: str
    callerid: str | None = None
    latching: int | None = None


def normalize_topic(name: str) -> str:
    if not name:
        return "/"
    return f'{"/" * (name[0] == "/")}{"/".join(x for x in name.split("/") if x)}'


def read_source_connection(record_header: Header, chunk: BytesIO) -> SourceConnection:
    conn_id = record_header.get_uint32("conn")
    topic = normalize_topic(record_header.get_string("topic"))
    data_header = Header.read(chunk)
    latching_raw = data_header.get_string("latching") if "latching" in data_header else None
    return SourceConnection(
        conn_id=conn_id,
        topic=topic,
        msgtype=normalize_msgtype(data_header.get_string("type")),
        msgdef=data_header.get_string("message_definition"),
        md5sum=data_header.get_string("md5sum"),
        callerid=data_header.get_string("callerid") if "callerid" in data_header else None,
        latching=int(latching_raw) if latching_raw not in (None, "") else None,
    )


def skip_record_data(src: BinaryIO | BytesIO) -> None:
    size = read_uint32(src)
    _ = src.seek(size, os.SEEK_CUR)


def read_padded_bag_header(src: BinaryIO) -> tuple[int, int, int]:
    magic = src.readline().decode(errors="replace")
    if not magic.startswith("#ROSBAG V2.0"):
        raise RuntimeError(f"unsupported or invalid bag magic: {magic.strip()!r}")
    header = Header.read(src, RecordType.BAGHEADER)
    index_pos = header.get_uint64("index_pos")
    conn_count = header.get_uint32("conn_count")
    chunk_count = header.get_uint32("chunk_count")
    pad_size = read_uint32(src)
    _ = src.seek(pad_size, os.SEEK_CUR)
    return index_pos, conn_count, chunk_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="source rosbag1 .bag")
    parser.add_argument("--dst", required=True, type=Path, help="destination rosbag2 directory")
    parser.add_argument(
        "--storage",
        choices=("sqlite3", "mcap"),
        default="sqlite3",
        help="rosbag2 storage backend",
    )
    parser.add_argument(
        "--include-topic",
        nargs="*",
        default=None,
        help="optional topic allowlist; default converts all topics",
    )
    parser.add_argument(
        "--exclude-topic",
        nargs="*",
        default=[],
        help="topics to skip",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="debug limit for quick validation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete existing destination before writing",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="print progress every N converted messages",
    )
    return parser.parse_args()


def should_convert(topic: str, include_topics: set[str] | None, exclude_topics: set[str]) -> bool:
    if topic in exclude_topics:
        return False
    return include_topics is None or topic in include_topics


def main() -> int:
    args = parse_args()
    if not args.src.exists():
        raise SystemExit(f"source bag not found: {args.src}")
    if args.dst.exists():
        if not args.overwrite:
            raise SystemExit(f"destination exists, pass --overwrite to replace it: {args.dst}")
        shutil.rmtree(args.dst)

    include_topics = set(args.include_topic) if args.include_topic else None
    exclude_topics = set(args.exclude_topic)
    storage = StoragePlugin.MCAP if args.storage == "mcap" else StoragePlugin.SQLITE3
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    source_connections: dict[int, SourceConnection] = {}
    writer_connections = {}
    counts: Counter[str] = Counter()
    chunk_count = 0
    converted = 0
    skipped_topics: Counter[str] = Counter()
    conversion_errors: Counter[str] = Counter()
    registered_custom_types: set[str] = set()

    with args.src.open("rb") as src, Rosbag2Writer(args.dst, version=8, storage_plugin=storage) as writer:
        index_pos, expected_connections, expected_chunks = read_padded_bag_header(src)
        print(
            "source header:",
            f"index_pos={index_pos}",
            f"conn_count={expected_connections}",
            f"chunk_count={expected_chunks}",
        )

        while True:
            if args.max_messages is not None and converted >= args.max_messages:
                break
            record_pos = src.tell()
            if index_pos and record_pos >= index_pos:
                break
            try:
                header = Header.read(src)
            except ReaderError as exc:
                print(f"stop: cannot read top-level record at byte {record_pos}: {exc}", file=sys.stderr)
                break

            try:
                op = header.get_uint8("op")
            except ReaderError as exc:
                print(f"stop: top-level record without op at byte {record_pos}: {exc}", file=sys.stderr)
                break

            if op != RecordType.CHUNK:
                if op == RecordType.CONNECTION:
                    _ = Header.read(src)
                else:
                    skip_record_data(src)
                continue

            compression = header.get_string("compression")
            data_size = read_uint32(src)
            raw_chunk = read_bytes(src, data_size)
            try:
                chunk_data = decompressors[compression](raw_chunk)
            except KeyError:
                raise SystemExit(f"unsupported rosbag1 compression {compression!r}") from None

            chunk_count += 1
            chunk = BytesIO(chunk_data)
            while chunk.tell() < len(chunk_data):
                if args.max_messages is not None and converted >= args.max_messages:
                    break
                inner_pos = chunk.tell()
                try:
                    inner_header = Header.read(chunk)
                    inner_op = inner_header.get_uint8("op")
                except ReaderError as exc:
                    print(
                        f"warning: stop chunk {chunk_count} at inner byte {inner_pos}: {exc}",
                        file=sys.stderr,
                    )
                    break

                if inner_op == RecordType.CONNECTION:
                    conn = read_source_connection(inner_header, chunk)
                    source_connections[conn.conn_id] = conn
                    msgtype = STATIC_MSGTYPE_RENAMES.get(conn.msgtype, conn.msgtype)
                    if msgtype not in typestore.fielddefs and msgtype not in registered_custom_types:
                        typs = get_types_from_msg(conn.msgdef, msgtype)
                        _ = typs.pop("std_msgs/msg/Header", None)
                        typestore.register(typs)
                        registered_custom_types.add(msgtype)
                    continue

                if inner_op != RecordType.MSGDATA:
                    skip_record_data(chunk)
                    continue

                conn_id = inner_header.get_uint32("conn")
                timestamp = inner_header.get_time("time")
                data = read_bytes(chunk, read_uint32(chunk))
                conn = source_connections.get(conn_id)
                if conn is None:
                    conversion_errors[f"unknown_conn_{conn_id}"] += 1
                    continue
                if not should_convert(conn.topic, include_topics, exclude_topics):
                    skipped_topics[conn.topic] += 1
                    continue

                msgtype = STATIC_MSGTYPE_RENAMES.get(conn.msgtype, conn.msgtype)
                key = (conn.conn_id, conn.topic, msgtype)
                if key not in writer_connections:
                    msgdef, _ = typestore.generate_msgdef(msgtype, ros_version=2)
                    writer_connections[key] = writer.add_connection(
                        conn.topic,
                        msgtype,
                        msgdef=msgdef,
                        rihs01=typestore.hash_rihs01(msgtype),
                        serialization_format="cdr",
                        offered_qos_profiles=DEFAULT_QOS,
                    )

                try:
                    cdr = typestore.ros1_to_cdr(data, typename=msgtype)
                except (SerdeError, KeyError, ValueError, AssertionError) as exc:
                    conversion_errors[f"{conn.topic}:{type(exc).__name__}"] += 1
                    continue

                writer.write(writer_connections[key], timestamp, cdr)
                counts[conn.topic] += 1
                converted += 1
                if args.progress_every and converted % args.progress_every == 0:
                    print(f"converted {converted} messages, chunk {chunk_count}/{expected_chunks}")

    print(f"wrote rosbag2: {args.dst}")
    print(f"chunks scanned: {chunk_count}")
    print(f"messages converted: {converted}")
    if counts:
        print("topic counts:")
        for topic, count in sorted(counts.items()):
            print(f"  {topic}: {count}")
    if skipped_topics:
        print("skipped topics:")
        for topic, count in sorted(skipped_topics.items()):
            print(f"  {topic}: {count}")
    if conversion_errors:
        print("conversion errors:")
        for key, count in sorted(conversion_errors.items()):
            print(f"  {key}: {count}")
    return 0 if converted else 2


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

import argparse
import sys

from moe_store.index import DEFAULT_PARTITION_SIZE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="moe-store")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser(
        "convert", help="convert a HF checkpoint into a v2 offload store"
    )
    convert.add_argument("checkpoint")
    convert.add_argument("store_dir")
    convert.add_argument(
        "--partition-size", type=int, default=DEFAULT_PARTITION_SIZE
    )

    inspect = sub.add_parser("inspect", help="print v2 store summary")
    inspect.add_argument("store_dir")
    inspect.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "convert":
        from moe_store.convert.convert import convert_checkpoint

        index = convert_checkpoint(
            args.checkpoint,
            args.store_dir,
            partition_size=args.partition_size,
        )
        print(
            f"converted {args.checkpoint}: {len(index.groups)} groups, "
            f"{index.num_tensors} tensors -> {args.store_dir}"
        )
        return 0

    if args.command == "inspect":
        from moe_store.convert.convert import dump_inspect

        print(dump_inspect(args.store_dir, verbose=args.verbose))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

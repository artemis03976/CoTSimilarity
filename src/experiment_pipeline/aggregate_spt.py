"""Aggregate SPT OOF predictions and compare against matched base and DSPR."""

from . import aggregate_lora


def main() -> int:
    return aggregate_lora.main(baseline_name="spt")


if __name__ == "__main__":
    raise SystemExit(main())

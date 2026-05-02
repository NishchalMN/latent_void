import argparse
import json
import sys

from latent_void.config import ConfigError, load_config, validate_config, with_overrides
from latent_void.pipeline import (
    discover_dataset,
    fuse_void,
    run_gsrecon,
    run_latent_inpaint,
    run_pipeline,
    run_segmentation,
    validate,
)


def _load(args):
    config = load_config(args.config)
    return with_overrides(config, getattr(args, "set", None))


def _print_json(data):
    print(json.dumps(data, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(prog="latent_void")
    subparsers = parser.add_subparsers(dest="command")

    def add_common(name):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--set", action="append", default=[], help="Override config key, e.g. project.output_dir=runs/x")
        return sub

    validate_parser = add_common("validate-config")
    validate_parser.add_argument("--strict-paths", action="store_true")

    add_common("discover-dataset")

    reconstruct = add_common("reconstruct")
    reconstruct.add_argument("--dry-run", action="store_true")

    segment = add_common("segment")
    segment.add_argument("--dry-run", action="store_true")

    add_common("fuse")

    inpaint = add_common("inpaint")
    inpaint.add_argument("--dry-run", action="store_true")

    run = add_common("run")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--skip-reconstruct", action="store_true")
    run.add_argument("--skip-segment", action="store_true")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2

    try:
        config = _load(args)
        if args.command == "validate-config":
            validate_config(config, strict_paths=args.strict_paths)
            result = validate(config, strict_paths=args.strict_paths)
        elif args.command == "discover-dataset":
            result = discover_dataset(config)
        elif args.command == "reconstruct":
            result = run_gsrecon(config, dry_run=args.dry_run)
        elif args.command == "segment":
            result = run_segmentation(config, dry_run=args.dry_run)
        elif args.command == "fuse":
            result = fuse_void(config)
        elif args.command == "inpaint":
            result = run_latent_inpaint(config, dry_run=args.dry_run)
        elif args.command == "run":
            result = run_pipeline(
                config,
                dry_run=args.dry_run,
                skip_reconstruct=args.skip_reconstruct,
                skip_segment=args.skip_segment,
            )
        else:
            parser.error("unknown command: %s" % args.command)
            return 2
        _print_json(result)
        return 0
    except (ConfigError, RuntimeError, ValueError, IOError) as exc:
        print("latent_void: error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

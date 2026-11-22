"""Build one fully evaluated immutable child population revision."""

from __future__ import annotations

import argparse
import json

from pokerl.training.population_evidence import build_child_population_revision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add frozen policy members to a population only after sampled, "
            "mirrored cross-play covers every unordered pair exactly once."
        )
    )
    parser.add_argument(
        "--parent",
        required=True,
        help="Existing immutable parent PopulationRevision JSON.",
    )
    parser.add_argument(
        "--member-spec",
        action="append",
        required=True,
        help="Exact PolicyMember JSON for a new policy; repeat as needed.",
    )
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        help="Completed evaluate_anchor JSON; repeat for every missing pair.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New path; existing different content is never overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revision = build_child_population_revision(
        parent_path=args.parent,
        member_spec_paths=args.member_spec,
        evidence_paths=args.evidence,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "revision_id": revision.revision_id,
                "parent_revision_id": revision.parent_revision_id,
                "policies": [member.policy_id for member in revision.policies],
                "evaluated_pairs": len(revision.payoffs),
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

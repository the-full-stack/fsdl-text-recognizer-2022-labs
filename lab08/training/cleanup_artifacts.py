"""Removes artifacts from projects and runs.

Artifacts are binary files that we want to track
and version but don't want to include in git,
generally because they are too large,
because they don't have meaningful diffs,
or because they change more quickly than code.

During development, we often generate artifacts
that we don't really need, e.g. model weights for
an overfitting test run. Space on artifact storage
is generally very large, but it is limited,
so we should occasionally delete unneeded artifacts
to reclaim some of that space.

For usage help, run
    python training/cleanup_artifacts.py --help
"""
import argparse

import wandb


api = wandb.Api()

DEFAULT_PROJECT = "fsdl-text-recognizer-2022-training"
DEFAULT_ENTITY = api.default_entity


def _setup_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="The entity from which to remove artifacts. Provide the value DEFAULT "
        + f"to use the default WANDB_ENTITY, which is currently {DEFAULT_ENTITY}.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help=f"The project from which to remove artifacts. Default is {DEFAULT_PROJECT}",
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        default=None,
        nargs="*",
        help="One or more run IDs from which to remove artifacts. Default is None.",
    )
    parser.add_argument(
        "--run_name_res",
        type=str,
        default=None,
        nargs="*",
        help="One or more regular expressions to use to select runs (by display name) from which to remove artifacts. See wandb.Api.runs documentation for details on the syntax. Beware that this is a footgun and consider using interactively with --dryrun and -v. Default is None.",
        metavar="RUN_NAME_REGEX",
    )

    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--all", action="store_true", help="Delete all artifacts from selected runs.")
    flags.add_argument(
        "--no-alias", action="store_true", help="Delete all artifacts without an alias from selected runs."
    )
    flags.add_argument(
        "--aliases",
        type=str,
        nargs="*",
        help="Delete artifacts that have any of the aliases from the provided list from selected runs.",
    )

    parser.add_argument(
        "-v",
        action="store_true",
        dest="verbose",
        help="Display information about targeted entities, projects, runs, and artifacts.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Select artifacts without deleting them and display which artifacts were selected.",
    )
    return parser


def main(args):
    entity = _get_entity_from(args)
    project_path = f"{entity}/{args.project}"

    runs = _get_runs(project_path, args.run_ids, args.run_name_res, verbose=args.verbose)
    artifact_selector = _get_selector_from(args)
    protect_aliases = args.no_alias  # avoid deletion of any aliased artifacts

    for run in runs:
        clean_run_artifacts(
            run, selector=artifact_selector, protect_aliases=protect_aliases, verbose=args.verbose, dryrun=args.dryrun
        )


def clean_run_artifacts(run, selector, protect_aliases=True, verbose=False, dryrun=True):
    artifacts = run.logged_artifacts()
    for artifact in artifacts:
        if selector(artifact):
            remove_artifact(artifact, protect_aliases=protect_aliases, verbose=verbose, dryrun=dryrun)


def remove_artifact(artifact, protect_aliases, verbose=False, dryrun=True):
    project, entity, id = artifact.project, artifact.entity, artifact.id
    type, aliases = artifact.type, artifact.aliases
    if verbose or dryrun:
        print(f"selecting for deletion artifact {project}/{entity}/{id} of type {type} with aliases {aliases}")
    if not dryrun:
        artifact.delete(delete_aliases=not protect_aliases)


def _get_runs(project_path, run_ids=None, run_name_res=None, verbose=False):
    if run_ids is None:
        run_ids = []

    if run_name_res is None:
        run_name_res = []

    runs = []
    for run_id in run_ids:
        runs.append(_get_run_by_id(project_path, run_id, verbose=verbose))

    for run_name_re in run_name_res:
        runs += _get_runs_by_name_re(project_path, run_name_re, verbose=verbose)

    return runs


def _get_run_by_id(project_path, run_id, verbose=False):
    path = f"{project_path}/{run_id}"
    run = api.run(path)
    if verbose:
        print(f"selecting run {run.entity}/{run.project}/{run.id} with display name {run.name}")
    return run


def _get_runs_by_name_re(project_path, run_name_re, verbose=False):
    matching_runs = api.runs(path=project_path, filters={"display_name": {"$regex": run_name_re}})

    if verbose:
        for run in matching_runs:
            print(f"selecting run {run.entity}/{run.project}/{run.id} with display name {run.name}")

    return matching_runs


def _get_selector_from(args, verbose=False):
    if args.all:
        if verbose:
            print("removing all artifacts from matching runs")
        return lambda _: True

    if args.no_alias:
        if verbose:
            print("removing all artifacts with no aliases from matching runs")
        return lambda artifact: artifact.aliases == []

    if args.aliases:
        if verbose:
            print(f"removing all artifacts with any of {args.aliases} in aliases from matching runs")
        return lambda artifact: any(alias in artifact.aliases for alias in args.aliases)

    if verbose:
        print("removing no artifacts matching runs")
    return lambda _: False


def _get_entity_from(args, verbose=False):
    entity = args.entity
    if entity is None:
        raise RuntimeError(f"No entity argument provided. Use --entity=DEFAULT to use {DEFAULT_ENTITY}.")
    elif entity == "DEFAULT":
        entity = DEFAULT_ENTITY
        if verbose:
            print(f"using default entity {entity}")
    else:
        if verbose:
            print(f"using entity {entity}")

    return entity


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    main(args)

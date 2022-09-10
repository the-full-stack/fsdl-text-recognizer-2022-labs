# Changelog & releases

The format of `CHANGELOG.md` is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Using towncrier to create news entries

We use [towncrier](https://github.com/twisted/towncrier) to reduce merge conflicts by generating
`CHANGELOG.md` from news fragments, rather than maintaining it directly. Create a news fragment for
each MR if you would like to ensure your changes are communicated to other project contributors.

```bash
# To create a news entry for an added feature relating to MR !123
# Adding --edit is optional and will open in your default shell's $EDITOR
towncrier create 123.added --edit
```

Top tips:
  - You may wish to add `export EDITOR="code -w"` to your `.zshrc` file to open this directly in VS Code.
  - News fragments should be written in markdown.
  - The generated news fragments live in `.changelog/` and can be easily rewritten as an MR evolves.

We use the following custom types (adapted from [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)):
  - `.added` for new features
  - `.changed` for changes in existing functionality
  - `.deprecated` for soon-to-be removed features
  - `.removed` for now removed features
  - `.fixed` for any bug fixes
  - `.security` in case of vulnerabilities
  - `.analysis` for data analyses
  - `.docs` for documentation improvements
  - `.maintenance` for maintenance tasks & upgrades


## Releasing a new version & updating `CHANGELOG.md`

Release versions are tied to Gitlab milestones and sprints. Release checklist:

1. Review MRs assigned to the release milestone in Gitlab & reallocate to the next release.
2. Run `towncrier build --version=VERSION` (preview with `--draft`)
3. Add a git tag for the release.

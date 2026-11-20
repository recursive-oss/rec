# Contributing to Recursive Engine

Thank you for your interest in `rec`. We welcome bug reports, documentation fixes,
tests, and code changes. This guide explains how to contribute.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By taking
part, you agree to follow it.

Do not share secrets in issues, pull requests, or logs. This includes API keys,
Hugging Face tokens, and private conversation content.

## Report a bug

Search the existing issues before you open a new one. Then open an issue with the
**Bug report** form. It asks for the information we need to diagnose the problem:
the output of `rec doctor` and `rec status`, your hardware and operating system,
the exact steps, and the full error.

Remove API keys, tokens, and private prompts before you post terminal output.
Malformed-request diagnostics can include the request body.

## Report a security vulnerability

Do not open a public issue for a security vulnerability. For private reporting
instructions, read the [security policy](SECURITY.md).

## Suggest a change

For a small fix, open a pull request directly. For a new feature, a new platform,
or a change to the command-line interface, release format, or model manifest, open
an issue first. Agreement on the approach before you write code saves time for
everyone.

## Set up a development environment

Before making a change, install the development requirements using
[Homebrew](https://brew.sh):

```sh
xcode-select --install
brew install go node uv
```

## Make a change

1. Fork the repository and create a branch from `main`.
2. Make one focused change. Unrelated changes belong in separate pull requests.
3. Add or update tests for changed behavior.
4. Update the documentation when you change commands, flags, output, or
   requirements.
5. Run `make test`.

## Quality standards and evaluation

The below standards are required for PRs intended for merge. Feel free to share
draft PRs that may not 100% fit the requirements;
please state clearly in the description that they are exploratory and not
intended for merge.

To be merged, contributions should meet a high standard of correctness, simplicity,
and maintainability. Please follow the existing architecture and coding style,
and keep added complexity to a minimum in terms of lines of code, files affected, and
conceptual complexity added.

AI tools are fine to use but you remain responsible for every line you submit.
Please review, understand, and test the full change, and be ready to explain
your choices.

Changes that could degrade performance require exhaustive evaluations. Include
reproducible before-and-after results across affected workloads and supported
hardware, covering latency, throughput, memory use, and output quality where
relevant. Please share the test setup and commands, and explain any regressions
or tradeoffs.

## Open a pull request

- Use a short, clear title.
- Describe what changed and why. Link the related issue.
- Tell us how you tested the change.
- Keep the pull request small enough to review.

We'll do our best to respond, but cannot guarantee a thorough review or a
timeline, especially for contributions that do not meet these requirements.
Review depends on maintainer availability and project priorities. A merge is not
guaranteed, and we may close PRs that fall outside the project's scope or
maintenance capacity.

## License

`rec` is licensed under the [Apache License, Version 2.0](LICENSE). By submitting a
contribution, you agree that it is licensed under the same terms, as Section 5 of
the license describes. You must have the right to submit the contribution.

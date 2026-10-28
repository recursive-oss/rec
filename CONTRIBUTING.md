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
5. Run `make test`. It runs the engine tests in parallel; set
   `PYTEST_XDIST_AUTO_NUM_WORKERS` to choose the worker count.

## Open a pull request

- Use a short, clear title.
- Describe what changed and why. Link the related issue.
- Tell us how you tested the change.
- Keep the pull request small enough to review.

A maintainer will review your pull request. We may ask for changes. We can decline
a change that does not fit the project's scope or that we cannot support.

## License

`rec` is licensed under the [Apache License, Version 2.0](LICENSE). By submitting a
contribution, you agree that it is licensed under the same terms, as Section 5 of
the license describes. You must have the right to submit the contribution.

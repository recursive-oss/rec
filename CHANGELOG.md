# Changelog

## v0.1.1

- Add support for Apple M4 Max.
- Add an Anthropic-compatible Messages API at `/v1/messages`, for Claude Code
  and the Anthropic SDK (#5).
- Derive tool call IDs from the request seed, so that a replayed request
  returns the same message (#3).

## v0.1.0

- Initial release.

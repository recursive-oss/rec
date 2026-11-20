# Changelog

## v0.1.2

- Add support for installing alongside another `rec` command.
- Restart a server that another version of rec started, so that `rec launch`
  and `rec serve` use the new version after an update.
- Add support for installing with Homebrew.
- Raise the HTTP server's listen backlog, so that a burst of requests gets the
  admission check's 429 instead of a reset connection (#27).

## v0.1.1

- Add support for Apple M4 Max.
- Add an Anthropic-compatible Messages API at `/v1/messages`, for Claude Code
  and the Anthropic SDK (#5).
- Derive tool call IDs from the request seed, so that a replayed request
  returns the same message (#3).

## v0.1.0

- Initial release.

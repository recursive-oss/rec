#!/usr/bin/env bash
# Keep invocation last: an incomplete transfer must not run an installation.
rec_install() (
  set -euo pipefail
  umask 077
  bold='' faint='' green='' red='' blue='' reset='' erase=''
  if [ -t 1 ] && [ "${TERM:-dumb}" != dumb ]; then
    erase=$'\r\033[K'
    if [ -z "${NO_COLOR:-}" ]; then
      bold=$'\033[1m' faint=$'\033[2m' green=$'\033[32m' red=$'\033[31m' blue=$'\033[38;5;68m' reset=$'\033[0m'
    fi
  fi
  fail() { printf '%s' "$erase"; printf '%srec install:%s %s\n' "$red$bold" "$reset" "$*" >&2; exit 1; }
  begin() { [ -z "$erase" ] || printf '  %s·%s %s' "$blue" "$reset" "$*"; }
  done_step() { printf '%s  %s✓%s %s\n' "$erase" "$green" "$reset" "$*"; }
  serve_after=0
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --serve) serve_after=1; shift;;
      --help)
        # shellcheck disable=SC2016 # Print the variables literally.
        printf 'Usage: install.sh [--serve]\nInstalls the latest rec release, or release tag v$REC_VERSION if set. --serve starts the Flint server afterwards.\n'; exit 0;;
      *) fail "unknown option: $1";;
    esac
  done
  [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || fail 'rec requires an Apple Silicon Mac'
  artifact=rec-darwin-arm64 variant=''
  # The default build's MLX runtime uses Metal 4, which needs macOS 26.
  macos=$(sw_vers -productVersion) || fail 'could not read the macOS version'
  [[ "$macos" =~ ^[0-9]+ ]] || fail "unrecognized macOS version: $macos"
  if [ "${BASH_REMATCH[0]}" -lt 26 ]; then variant=-macos15; artifact=$artifact$variant; fi
  version=${REC_VERSION:-}
  version=${version#v}
  [ -z "$version" ] || [[ "$version" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail 'invalid REC_VERSION'
  releases=${REC_RELEASES_URL:-https://github.com/recursive-oss/rec/releases}
  proto='=https'
  case "$releases" in
    https://*) ;;
    http://127.0.0.1:*) [ "${REC_INSTALL_ALLOW_HTTP:-}" = 1 ] || fail 'HTTP requires REC_INSTALL_ALLOW_HTTP=1 for local tests'; proto='=http';;
    *) fail 'REC_RELEASES_URL must use HTTPS';;
  esac
  data=${XDG_DATA_HOME:-$HOME/.local/share}/rec
  bin="$HOME/.local/bin"
  case "$data" in /*) ;; *) fail 'XDG_DATA_HOME must be absolute';; esac
  mkdir -p "$data/releases" "$bin"
  chmod 700 "$data"
  lock_dir="$data/recursive_rec.lock"
  mkdir "$lock_dir" 2>/dev/null || fail "another install is active; if interrupted, remove $lock_dir"
  tmp=$(mktemp -d "$data/.install.XXXXXX")
  trap 'rm -rf "$tmp"; rmdir "$lock_dir" 2>/dev/null || true' EXIT
  existing=$(command -v rec || true)
  if [ -e "$bin/rec" ] || [ -L "$bin/rec" ]; then
    [ -L "$bin/rec" ] || fail "$bin/rec exists and is not a managed symlink"
    old=$(readlink "$bin/rec")
    case "$old" in "$data"/releases/*/rec) [ -f "$(dirname "$old")/.managed" ] || fail 'existing rec is not managed';; *) fail 'existing rec belongs to another program';; esac
    [ -z "$existing" ] || [ "$existing" = "$bin/rec" ] || fail "rec already resolves to $existing"
  else
    old=''
    [ -z "$existing" ] || fail "rec already resolves to $existing"
  fi
  printf '%sInstalling rec%s\n\n' "$bold" "$reset"
  begin "Downloading rec ${version:-(latest release)}"
  command -v curl >/dev/null || fail 'curl is required'
  if [ -n "$version" ]; then path="download/v$version"; else path=latest/download; fi
  for name in "$artifact" "$artifact.meta"; do
    curl --proto "$proto" --proto-redir "$proto" --tlsv1.2 --fail --silent --show-error --location --retry 3 --connect-timeout 15 \
      "$releases/$path/$name" -o "$tmp/$name" || fail "download failed: $releases/$path/$name"
  done
  read -r expected_hash expected_size extra < "$tmp/$artifact.meta"
  [[ "$expected_hash" =~ ^[a-f0-9]{64}$ ]] && [[ "$expected_size" =~ ^[0-9]+$ ]] && [ -z "$extra" ] || fail 'invalid release metadata'
  actual_hash=$(shasum -a 256 "$tmp/$artifact"); actual_hash=${actual_hash%% *}
  actual_size=$(wc -c < "$tmp/$artifact" | tr -d '[:space:]')
  [ "$actual_hash" = "$expected_hash" ] && [ "$actual_size" = "$expected_size" ] || fail 'binary checksum or size mismatch'
  chmod 700 "$tmp/$artifact"
  reported=$("$tmp/$artifact" --version) || fail 'downloaded binary does not run'
  reported=${reported#rec }
  [[ "$reported" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail 'binary version check failed'
  [ -z "$version" ] || [ "$reported" = "$version" ] || fail "binary reports version $reported, expected $version"
  version=$reported
  done_step "Downloaded rec $version"
  begin 'Checking this Mac'
  doctor=$("$tmp/$artifact" doctor 2>&1) || { printf '%s' "$erase"; printf '%s\n' "$doctor" >&2; fail 'this Mac does not meet the requirements above'; }
  done_step 'This Mac meets the requirements'
  release="$data/releases/$version$variant"
  if [ -e "$release" ]; then
    [ -f "$release/.managed" ] && [ -f "$release/rec" ] || fail 'release directory is not managed'
    installed_hash=$(shasum -a 256 "$release/rec"); installed_hash=${installed_hash%% *}
    [ "$installed_hash" = "$expected_hash" ] || fail 'immutable release version has different bytes'
  else
    mkdir "$tmp/release"
    mv "$tmp/$artifact" "$tmp/release/rec"
    printf '%s\n' "$expected_hash" > "$tmp/release/.managed"
    mv "$tmp/release" "$release"
  fi
  if [ -n "$old" ] && "$old" status >/dev/null 2>&1; then
    begin 'Stopping the running model'
    "$old" stop >/dev/null || fail 'could not stop the running model; run rec stop and retry'
    for _ in $(seq 150); do "$old" status >/dev/null 2>&1 || break; sleep 0.2; done
    ! "$old" status >/dev/null 2>&1 || fail 'the running model did not stop; run rec stop and retry'
    done_step 'Stopped the running model'
  fi
  [ -z "$old" ] || printf '%s\n' "$old" > "$data/previous"
  link_dir=$(mktemp -d "$bin/.rec-link.XXXXXX")
  ln -s "$release/rec" "$link_dir/rec"
  mv -f "$link_dir/rec" "$bin/rec"
  rmdir "$link_dir"
  add_path() {
    file=$1
    if ! grep -Fq '# Recursive CLI PATH' "$file" 2>/dev/null; then
      # shellcheck disable=SC2016 # Print the variables literally.
      printf '\n# Recursive CLI PATH\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$file"
    fi
  }
  case "${SHELL:-}" in
    */zsh) add_path "${ZDOTDIR:-$HOME}/.zshrc";;
    */bash)
      add_path "$HOME/.bashrc"
      if [ -f "$HOME/.bash_profile" ]; then add_path "$HOME/.bash_profile"; elif [ -f "$HOME/.bash_login" ]; then add_path "$HOME/.bash_login"; else add_path "$HOME/.profile"; fi;;
    *) shell_note='Add ~/.local/bin to PATH in your shell configuration.';;
  esac
  done_step "Installed rec $version in ~/.local/bin"
  printf '\n%sGet started%s\n' "$bold" "$reset"
  printf '  %s%-21s%s %s\n' \
    "$blue" 'rec launch pi' "$reset" 'Chat with Flint in pi' \
    "$blue" 'rec launch opencode' "$reset" 'Chat with Flint in OpenCode' \
    "$blue" 'rec serve' "$reset" 'Serve the local API for your own client'
  if [ -n "${shell_note:-}" ]; then
    printf '\n%s%s%s\n' "$faint" "$shell_note" "$reset"
  else
    # shellcheck disable=SC2016 # Print the variables literally.
    case ":$PATH:" in
      *":$bin:"*) ;;
      *) printf '\n%sOpen a new terminal, or run: export PATH="$HOME/.local/bin:$PATH"%s\n' "$faint" "$reset";;
    esac
  fi
  if [ "$serve_after" = 1 ]; then
    rm -rf "$tmp"
    rmdir "$lock_dir"
    trap - EXIT
    "$bin/rec" serve
  fi
)
rec_install "$@"

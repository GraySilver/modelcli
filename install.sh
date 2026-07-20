#!/bin/sh

set -eu

: "${HOME:?HOME must be set}"

REPOSITORY=${MODELCLI_REPOSITORY:-https://github.com/GraySilver/modelcli.git}
REF=${MODELCLI_REF:-main}
PYTHON_VERSION=${MODELCLI_PYTHON:-3.12}
DATA_HOME=${XDG_DATA_HOME:-"$HOME/.local/share"}
INSTALL_DIR=${MODELCLI_INSTALL_DIR:-"$DATA_HOME/modelcli"}
BIN_DIR=${MODELCLI_BIN_DIR:-"$HOME/.local/bin"}
UV_BIN=${MODELCLI_UV_BIN:-}
COMMAND_PATH="$BIN_DIR/modelcli"
ENTRY_POINT="$INSTALL_DIR/.venv/bin/modelcli"
MARKER_PATH="$INSTALL_DIR/.git/modelcli-installer"
UV_TMP=

say() {
    printf '%s\n' "[modelcli] $*"
}

die() {
    printf '%s\n' "[modelcli] error: $*" >&2
    exit 1
}

cleanup() {
    if [ -n "$UV_TMP" ] && [ -d "$UV_TMP" ]; then
        rm -rf "$UV_TMP"
    fi
}

trap cleanup 0

usage() {
    cat <<'EOF'
Install or update ModelCLI.

Usage:
  install.sh
  install.sh --uninstall
  install.sh --help

Environment:
  MODELCLI_PYTHON       Python version for the environment (default: 3.12)
  MODELCLI_REPOSITORY   Git repository URL
  MODELCLI_REF          Git branch or tag (default: main)
  MODELCLI_INSTALL_DIR  Managed checkout location
  MODELCLI_BIN_DIR      Directory for the modelcli command
  MODELCLI_UV_BIN       Path to an existing uv executable
EOF
}

validate_paths() {
    case "$INSTALL_DIR" in
        ""|/|"$HOME")
            die "refusing unsafe install directory: $INSTALL_DIR"
            ;;
    esac
    case "$BIN_DIR" in
        ""|/)
            die "refusing unsafe command directory: $BIN_DIR"
            ;;
    esac
}

uninstall_modelcli() {
    validate_paths

    if [ -L "$COMMAND_PATH" ]; then
        target=$(readlink "$COMMAND_PATH" 2>/dev/null || true)
        if [ "$target" = "$ENTRY_POINT" ]; then
            rm -f "$COMMAND_PATH"
            say "Removed $COMMAND_PATH"
        else
            say "Left unrelated symlink untouched: $COMMAND_PATH"
        fi
    elif [ -e "$COMMAND_PATH" ]; then
        say "Left unrelated command untouched: $COMMAND_PATH"
    fi

    if [ -d "$INSTALL_DIR" ]; then
        [ -f "$MARKER_PATH" ] || die "$INSTALL_DIR is not managed by this installer"
        rm -rf "$INSTALL_DIR"
        say "Removed $INSTALL_DIR"
    else
        say "ModelCLI is not installed in $INSTALL_DIR"
    fi

    say "Model caches were kept. Run 'modelcli models clean' before uninstalling to remove them."
}

install_uv() {
    UV_TMP=$(mktemp -d "${TMPDIR:-/tmp}/modelcli-uv.XXXXXX")
    installer="$UV_TMP/install.sh"

    say "uv was not found; downloading the official installer"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh -o "$installer"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$installer" https://astral.sh/uv/install.sh
    else
        die "curl or wget is required to install uv"
    fi

    env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh "$installer"
    UV_BIN="$HOME/.local/bin/uv"
    [ -x "$UV_BIN" ] || die "uv installation did not create $UV_BIN"
}

find_uv() {
    if [ -n "$UV_BIN" ]; then
        [ -x "$UV_BIN" ] || die "MODELCLI_UV_BIN is not executable: $UV_BIN"
        return
    fi
    if command -v uv >/dev/null 2>&1; then
        UV_BIN=$(command -v uv)
        return
    fi
    install_uv
}

check_command_path() {
    if [ -e "$COMMAND_PATH" ] || [ -L "$COMMAND_PATH" ]; then
        target=$(readlink "$COMMAND_PATH" 2>/dev/null || true)
        [ "$target" = "$ENTRY_POINT" ] || die "$COMMAND_PATH already exists and is not managed by this installer"
    fi
}

checkout_source() {
    command -v git >/dev/null 2>&1 || die "git is required"

    if [ -e "$INSTALL_DIR" ]; then
        [ -d "$INSTALL_DIR/.git" ] || die "$INSTALL_DIR exists but is not a Git checkout"
        [ -f "$MARKER_PATH" ] || die "$INSTALL_DIR is not managed by this installer"
        git -C "$INSTALL_DIR" diff --quiet || die "$INSTALL_DIR contains local changes"
        git -C "$INSTALL_DIR" diff --cached --quiet || die "$INSTALL_DIR contains staged changes"
        current_repository=$(git -C "$INSTALL_DIR" remote get-url origin)
        [ "$current_repository" = "$REPOSITORY" ] || die "$INSTALL_DIR uses a different origin: $current_repository"

        say "Updating source from $REPOSITORY ($REF)"
        git -C "$INSTALL_DIR" fetch --depth 1 origin "$REF"
        git -C "$INSTALL_DIR" checkout --detach --quiet FETCH_HEAD
    else
        say "Cloning $REPOSITORY ($REF)"
        mkdir -p "$(dirname "$INSTALL_DIR")"
        git clone --depth 1 --branch "$REF" "$REPOSITORY" "$INSTALL_DIR"
        git -C "$INSTALL_DIR" checkout --detach --quiet
        : > "$MARKER_PATH"
    fi
}

install_modelcli() {
    validate_paths
    check_command_path
    find_uv
    checkout_source

    say "Creating the locked Python $PYTHON_VERSION environment"
    UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/.venv" \
        "$UV_BIN" sync --directory "$INSTALL_DIR" --frozen --no-dev --python "$PYTHON_VERSION"
    [ -x "$ENTRY_POINT" ] || die "installation completed without creating $ENTRY_POINT"

    mkdir -p "$BIN_DIR"
    ln -sfn "$ENTRY_POINT" "$COMMAND_PATH"

    version=$("$COMMAND_PATH" --version)
    say "Installed ModelCLI $version"
    say "Command: $COMMAND_PATH"
    say "Models are downloaded only when requested. Run 'modelcli models list' to inspect them."

    case ":${PATH:-}:" in
        *":$BIN_DIR:"*)
            say "Run 'modelcli --help' to get started."
            ;;
        *)
            say "Add this line to your shell profile: export PATH=\"$BIN_DIR:\$PATH\""
            ;;
    esac
}

case ${1:-} in
    "")
        [ "$#" -eq 0 ] || die "unexpected arguments"
        install_modelcli
        ;;
    --uninstall)
        [ "$#" -eq 1 ] || die "--uninstall does not accept arguments"
        uninstall_modelcli
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage >&2
        die "unknown option: $1"
        ;;
esac

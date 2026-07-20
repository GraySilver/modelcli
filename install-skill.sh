#!/bin/sh

set -eu

: "${HOME:?HOME must be set}"

REPOSITORY=${MODELCLI_REPOSITORY:-https://github.com/GraySilver/modelcli.git}
REF=${MODELCLI_REF:-main}
AGENTS_SKILLS_DIR=${MODELCLI_AGENTS_SKILLS_DIR:-"$HOME/.agents/skills"}
CLAUDE_SKILLS_DIR=${MODELCLI_CLAUDE_SKILLS_DIR:-"$HOME/.claude/skills"}
TARGET=all
UNINSTALL=0
SOURCE_DIR=
TEMP_DIR=
STAGE_DIR=
MARKER=.modelcli-skill-installer

say() {
    printf '%s\n' "[modelcli-skill] $*"
}

die() {
    printf '%s\n' "[modelcli-skill] error: $*" >&2
    exit 1
}

cleanup() {
    if [ -n "$STAGE_DIR" ] && [ -e "$STAGE_DIR" ]; then
        rm -rf "$STAGE_DIR"
    fi
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

trap cleanup 0

usage() {
    cat <<'EOF'
Install the ModelCLI Agent Skill for Codex, Claude Code, or OpenClaw.

Usage:
  install-skill.sh [--target all|codex|claude|openclaw]
  install-skill.sh [--target all|codex|claude|openclaw] --uninstall
  install-skill.sh --help

The default target is all. Codex and OpenClaw share ~/.agents/skills/modelcli;
Claude Code uses ~/.claude/skills/modelcli.

Environment:
  MODELCLI_REPOSITORY          Git repository URL used by piped installs
  MODELCLI_REF                 Git branch or tag (default: main)
  MODELCLI_AGENTS_SKILLS_DIR   Override the Codex/OpenClaw skills directory
  MODELCLI_CLAUDE_SKILLS_DIR   Override the Claude Code skills directory
  MODELCLI_SKILL_SOURCE        Override the local skills/modelcli source directory
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --target)
            [ "$#" -ge 2 ] || die "--target requires a value"
            TARGET=$2
            shift 2
            ;;
        --uninstall)
            UNINSTALL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "unknown option: $1"
            ;;
    esac
done

case "$TARGET" in
    all|codex|claude|openclaw) ;;
    *) die "unknown target: $TARGET" ;;
esac

validate_destination() {
    destination=$1
    case "$destination" in
        ""|/|"$HOME") die "refusing unsafe skill destination: $destination" ;;
        */modelcli) ;;
        *) die "skill destination must end in /modelcli: $destination" ;;
    esac
}

uninstall_skill() {
    destination=$1
    label=$2
    validate_destination "$destination"

    if [ -L "$destination" ]; then
        die "$destination is a symlink and is not managed by this installer"
    fi
    if [ ! -e "$destination" ]; then
        say "$label skill is not installed in $destination"
        return
    fi
    [ -d "$destination" ] || die "$destination exists and is not a directory"
    [ -f "$destination/$MARKER" ] || die "$destination is not managed by this installer"
    rm -rf "$destination"
    say "Removed $label skill from $destination"
}

install_skill() {
    destination=$1
    label=$2
    validate_destination "$destination"

    if [ -L "$destination" ]; then
        die "$destination is a symlink and will not be overwritten"
    fi
    if [ -e "$destination" ]; then
        [ -d "$destination" ] || die "$destination exists and is not a directory"
        [ -f "$destination/$MARKER" ] || die "$destination is not managed by this installer"
    fi

    parent=$(dirname "$destination")
    mkdir -p "$parent"
    STAGE_DIR=$(mktemp -d "$parent/.modelcli-skill.XXXXXX")
    cp -R "$SOURCE_DIR/." "$STAGE_DIR/"
    [ -f "$STAGE_DIR/SKILL.md" ] || die "skill source does not contain SKILL.md"
    printf '%s\n' "Installed from $REPOSITORY at $REF" > "$STAGE_DIR/$MARKER"

    if [ -d "$destination" ]; then
        rm -rf "$destination"
    fi
    mv "$STAGE_DIR" "$destination"
    STAGE_DIR=
    say "Installed $label skill in $destination"
}

use_agents_target=0
use_claude_target=0
case "$TARGET" in
    all)
        use_agents_target=1
        use_claude_target=1
        ;;
    codex|openclaw)
        use_agents_target=1
        ;;
    claude)
        use_claude_target=1
        ;;
esac

agents_destination="$AGENTS_SKILLS_DIR/modelcli"
claude_destination="$CLAUDE_SKILLS_DIR/modelcli"

if [ "$UNINSTALL" -eq 1 ]; then
    if [ "$use_agents_target" -eq 1 ]; then
        uninstall_skill "$agents_destination" "Codex/OpenClaw"
    fi
    if [ "$use_claude_target" -eq 1 ]; then
        uninstall_skill "$claude_destination" "Claude Code"
    fi
    exit 0
fi

if [ -n "${MODELCLI_SKILL_SOURCE:-}" ]; then
    SOURCE_DIR=$MODELCLI_SKILL_SOURCE
elif [ -f "$0" ]; then
    script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
    if [ -f "$script_dir/skills/modelcli/SKILL.md" ]; then
        SOURCE_DIR="$script_dir/skills/modelcli"
    fi
fi

if [ -z "$SOURCE_DIR" ]; then
    command -v git >/dev/null 2>&1 || die "git is required for a remote skill install"
    TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/modelcli-skill.XXXXXX")
    say "Downloading skill from $REPOSITORY ($REF)"
    git clone --depth 1 --branch "$REF" --quiet "$REPOSITORY" "$TEMP_DIR/repository"
    SOURCE_DIR="$TEMP_DIR/repository/skills/modelcli"
fi

[ -f "$SOURCE_DIR/SKILL.md" ] || die "ModelCLI skill was not found in $SOURCE_DIR"

if [ "$use_agents_target" -eq 1 ]; then
    install_skill "$agents_destination" "Codex/OpenClaw"
fi
if [ "$use_claude_target" -eq 1 ]; then
    install_skill "$claude_destination" "Claude Code"
fi

say "Restart the selected agent so it can discover the skill."

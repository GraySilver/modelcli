from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "install-skill.sh"


def run_installer(home: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    return subprocess.run(
        ["sh", str(INSTALLER), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_installer_installs_updates_and_uninstalls_all_targets(tmp_path: Path) -> None:
    home = tmp_path / "home"
    agents = home / ".agents" / "skills" / "modelcli"
    claude = home / ".claude" / "skills" / "modelcli"

    installed = run_installer(home)
    assert installed.returncode == 0, installed.stderr
    for destination in (agents, claude):
        assert (destination / "SKILL.md").is_file()
        assert (destination / "agents" / "openai.yaml").is_file()
        assert (destination / ".modelcli-skill-installer").is_file()
        assert not destination.is_symlink()

    stale = agents / "stale.txt"
    stale.write_text("old")
    updated = run_installer(home, "--target", "codex")
    assert updated.returncode == 0, updated.stderr
    assert not stale.exists()

    removed = run_installer(home, "--target", "all", "--uninstall")
    assert removed.returncode == 0, removed.stderr
    assert not agents.exists()
    assert not claude.exists()


def test_installer_does_not_overwrite_or_remove_unowned_skill(tmp_path: Path) -> None:
    home = tmp_path / "home"
    destination = home / ".claude" / "skills" / "modelcli"
    destination.mkdir(parents=True)
    original = destination / "SKILL.md"
    original.write_text("user-owned")

    install = run_installer(home, "--target", "claude")
    assert install.returncode != 0
    assert "not managed by this installer" in install.stderr
    assert original.read_text() == "user-owned"

    uninstall = run_installer(home, "--target", "claude", "--uninstall")
    assert uninstall.returncode != 0
    assert "not managed by this installer" in uninstall.stderr
    assert original.read_text() == "user-owned"


def test_piped_installer_clones_skill_source(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    skill_source = repository / "skills" / "modelcli"
    skill_source.parent.mkdir(parents=True)
    shutil.copytree(ROOT / "skills" / "modelcli", skill_source)
    subprocess.run(["git", "init", "-b", "main", str(repository)], capture_output=True, text=True, check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=ModelCLI Tests",
            "-c",
            "user.email=modelcli-tests@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    home = tmp_path / "home"
    env = os.environ.copy()
    env.update({"HOME": str(home), "MODELCLI_REPOSITORY": str(repository)})
    result = subprocess.run(
        ["sh", "-s", "--", "--target", "claude"],
        cwd=tmp_path,
        input=INSTALLER.read_text(),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    destination = home / ".claude" / "skills" / "modelcli"
    assert result.returncode == 0, result.stderr
    assert "Downloading skill" in result.stdout
    assert (destination / "SKILL.md").is_file()
    assert (destination / ".modelcli-skill-installer").is_file()

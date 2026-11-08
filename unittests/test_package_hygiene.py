import subprocess
import sys

import svdynamics


def test_import_does_not_modify_builtin_help():
    code = (
        "import builtins; h = builtins.help; "
        "import svdynamics; "
        "assert builtins.help is h, 'builtins.help was replaced'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_package_help_prints_logo_and_overview(capsys):
    svdynamics.help()
    out = capsys.readouterr().out
    assert svdynamics.svdynamics_logo.strip() in out
    assert "Welcome to svdynamics" in out

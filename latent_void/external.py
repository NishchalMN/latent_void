import os
import shlex
import subprocess

from latent_void.io import ensure_dir, write_json


class ExternalCommandError(RuntimeError):
    pass


def format_command(template, values):
    return template.format(**values)


def run_command(template, values, dry_run=False, cwd=None):
    if not template:
        raise ExternalCommandError("external command template is empty")
    command = format_command(template, values)
    if dry_run:
        return {"dry_run": True, "command": command, "returncode": None}
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    output_lines = []
    for line in proc.stdout:
        print(line.rstrip())
        output_lines.append(line)
    returncode = proc.wait()
    if returncode != 0:
        raise ExternalCommandError("command failed with code %s: %s" % (returncode, command))
    return {"dry_run": False, "command": command, "returncode": returncode, "output": "".join(output_lines)}


def write_views_manifest(path, views, extra=None):
    ensure_dir(os.path.dirname(path))
    payload = {"views": [view.to_manifest() for view in views]}
    if extra:
        payload.update(extra)
    write_json(path, payload)
    return path


def shell_quote(value):
    return shlex.quote(str(value))

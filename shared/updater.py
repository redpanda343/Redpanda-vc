import os
import shutil
import subprocess
import sys


def pull_updates(directory, app_name):
    if shutil.which("git") is None:
        print("Git is unavailable; starting the installed version.", flush=True)
        return False
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")

    def git(*arguments, check=True):
        command = ["git"]
        if os.name == "nt":
            command.extend(["-c", "http.sslBackend=openssl"])
        result = subprocess.run(
            [*command, *arguments], cwd=directory, env=environment,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        if check and result.returncode:
            message = (result.stderr + result.stdout).strip()
            raise RuntimeError(message or f"Git exited with code {result.returncode}.")
        return result

    root = git("rev-parse", "--show-toplevel", check=False)
    if root.returncode or os.path.normcase(os.path.realpath(root.stdout.strip())) != os.path.normcase(os.path.realpath(directory)):
        print("No Git checkout at the app folder; starting the installed version.", flush=True)
        return False
    if git("diff", "--name-only", "--diff-filter=U").stdout.strip():
        raise SystemExit(f"Resolve the existing Git conflicts before starting the {app_name}.")
    if git("branch", "--show-current").stdout.strip() != "main":
        print("Automatic updates apply to main checkouts only; starting the current branch.", flush=True)
        return False
    previous = git("rev-parse", "HEAD").stdout.strip()
    print("Checking origin/main for updates...", flush=True)
    result = git("pull", "--ff-only", "--autostash", "origin", "main", check=False)
    message = (result.stdout + result.stderr).strip()
    if message:
        print(message, flush=True)
    if git("diff", "--name-only", "--diff-filter=U").stdout.strip():
        raise SystemExit(f"Git could not reapply local edits cleanly. Resolve the conflicts before starting the {app_name}.")
    if result.returncode:
        raise RuntimeError(f"Update failed with Git exit code {result.returncode}.")
    return git("rev-parse", "HEAD").stdout.strip() != previous


def update_and_restart(script_path, app_name):
    script_path = os.path.abspath(script_path)
    os.chdir(os.path.dirname(script_path))
    if os.environ.pop("REDPANDA_EXPERIMENTAL_UPDATE_RESTART", "") == script_path:
        return
    try:
        if pull_updates(os.path.dirname(script_path), app_name):
            os.environ["REDPANDA_EXPERIMENTAL_UPDATE_RESTART"] = script_path
            os.execv(sys.executable, [sys.executable, script_path, *sys.argv[1:]])
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"Automatic update skipped: {error}. Starting the installed version.", file=sys.stderr, flush=True)

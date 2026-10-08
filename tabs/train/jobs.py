import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import psutil

from rectified_flow.distributed import parse_devices

ROOT = Path(__file__).resolve().parents[2]
QUIET_WARNINGS = ('ignore:pkg_resources is deprecated,ignore:Checkpoint directory,'
                  'ignore:`isinstance(treespec,ignore::FutureWarning')


def experiment_path(name):
    name = str(name).strip()
    if not name or Path(name).name != name or name in {'.', '..'} or any(c in name for c in '/\\:'):
        raise gr.Error('Enter a model name without folders or path separators.')
    return ROOT / 'logs' / name


def positive_integer(value, label):
    if value is None or float(value) != int(value) or int(value) < 1:
        raise gr.Error(f'{label} must be a positive whole number.')
    return str(int(value))


def device_id(device):
    try:
        devices = parse_devices(device)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return '-' if devices == ['cpu'] else '-'.join(item[5:] for item in devices)


def terminate_tree(process):
    try:
        parent = psutil.Process(process.pid)
        processes = parent.children(recursive=True) + [parent]
        for child in processes:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=3)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.wait(timeout=5)
    except psutil.NoSuchProcess:
        pass


class JobRunner:
    def __init__(self, kind):
        self.kind = kind
        self._lock = threading.Lock()
        self._process = None
        self._runner = None
        self._cancelled = False

    def _active(self):
        return self._runner is not None and self._runner.is_alive()

    def _controls(self, idle):
        active = self._active()
        return (*[gr.update(interactive=not active) for _ in range(idle)], gr.update(interactive=active),
                gr.Timer(active=active))

    def controls(self, idle):
        with self._lock:
            return self._controls(idle)

    def _run(self, label, steps, finished):
        warnings = ','.join(filter(None, (os.environ.get('PYTHONWARNINGS'), QUIET_WARNINGS)))
        environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONWARNINGS=warnings)
        for module, arguments in steps:
            with self._lock:
                if self._cancelled:
                    return
                self._process = subprocess.Popen([sys.executable, '-u', '-m', module, *arguments], cwd=ROOT,
                                                 env=environment)
            code = self._process.wait()
            if self._cancelled:
                return
            if code != 0:
                print(f'{label} stopped with exit code {code}.', flush=True)
                return
        print(f'{label} {finished}', flush=True)

    def launch(self, name, steps, description, idle, finished='done.'):
        directory = experiment_path(name)
        with self._lock:
            if self._active():
                raise gr.Error(f'A {self.kind} job is already running. Wait for it to finish or stop it first.')
            directory.mkdir(parents=True, exist_ok=True)
            self._cancelled = False
            label = f'{description} {name}:'
            print(f'{label} started.', flush=True)
            self._runner = threading.Thread(target=self._run, args=(label, steps, finished), daemon=True)
            self._runner.start()
            gr.Info(f'{description} started. Progress is shown in the console.')
            return self._controls(idle)

    def stop(self, idle):
        with self._lock:
            self._cancelled = True
            if self._process is not None and self._process.poll() is None:
                terminate_tree(self._process)
                print('Stopped. Resume starts from the last saved checkpoint; unsaved steps are lost.', flush=True)
            return self._controls(idle)

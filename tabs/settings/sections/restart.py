import os
import sys

import gradio as gr
import psutil

from rvc.train.process.training_control import (
    get_training_state,
    pause_training,
    resume_training,
    stop_training,
)

now_dir = os.getcwd()


def stop_train(model_name: str):
    return stop_training(os.path.join(now_dir, "logs"), model_name)


def pause_train(model_name: str):
    return pause_training(os.path.join(now_dir, "logs"), model_name)


def resume_train(model_name: str):
    return resume_training(os.path.join(now_dir, "logs"), model_name)


def get_train_state(model_name: str):
    return get_training_state(os.path.join(now_dir, "logs"), model_name)


def stop_infer():
    pid_file_path = os.path.join(now_dir, "assets", "infer_pid.txt")
    try:
        with open(pid_file_path, "r") as pid_file:
            pids = [int(pid) for pid in pid_file.readlines() if pid.strip()]

        for pid in pids:
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception:
                try:
                    os.kill(pid, 9)
                except:
                    pass

        os.remove(pid_file_path)
    except:
        pass


def restart_applio():
    if os.name != "nt":
        os.system("clear")
    else:
        os.system("cls")
    python = sys.executable
    os.execl(python, python, *sys.argv)


from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def restart_tab():
    with gr.Row():
        with gr.Column():
            restart_button = gr.Button(i18n("Restart redpanda-vc"))
            restart_button.click(
                fn=restart_applio,
                inputs=[],
                outputs=[],
            )

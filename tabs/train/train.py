import gradio as gr

from tabs.train.rvc import rvc_train_tab


def train_tab():
    with gr.Tab("RVC"):
        rvc_train_tab()

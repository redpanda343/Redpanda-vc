import gradio as gr

from tabs.train.causalvc import causalvc_train_tab
from tabs.train.rvc import rvc_train_tab


def train_tab():
    with gr.Tab("RVC"):
        rvc_train_tab()
    with gr.Tab("Causal Voice Changer"):
        causalvc_train_tab()

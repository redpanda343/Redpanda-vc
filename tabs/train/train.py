import gradio as gr

from tabs.train.beatrice import beatrice_train_tab
from tabs.train.nsf_hifigan import nsf_hifigan_train_tab
from tabs.train.rectified_flow import rectified_train_tab
from tabs.train.rvc import rvc_train_tab


def train_tab():
    with gr.Tab("RVC"):
        rvc_train_tab()
    with gr.Tab("Rectified Flow"):
        rectified_train_tab()
    with gr.Tab("NSF-HiFiGAN Vocoder"):
        nsf_hifigan_train_tab()
    with gr.Tab("Beatrice"):
        beatrice_train_tab()

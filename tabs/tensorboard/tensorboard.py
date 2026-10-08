import gradio as gr

from assets.i18n.i18n import I18nAuto
from shared.tools.launch_tensorboard import launch_tensorboard

i18n = I18nAuto()


def tensorboard_tab():
    def launch_and_embed():
        url = launch_tensorboard()
        if url:
            gr.Info(i18n("TensorBoard ready."))
            return (
                '<iframe src="/tensorboard/" title="TensorBoard" '
                'style="width:100%;height:800px;border:0" '
                'referrerpolicy="same-origin"></iframe>'
            )
        gr.Warning(
            i18n("TensorBoard could not be started. Check the console for details.")
        )
        return "<p>TensorBoard could not be started.</p>"

    with gr.Column():
        gr.Markdown(i18n("### TensorBoard\nMonitor training metrics as they update."))
        with gr.Row():
            launch_btn = gr.Button(i18n("Launch TensorBoard"), variant="primary")
        tb_iframe = gr.HTML(
            value=(
                "<p style='color: gray; text-align: center; padding: 40px;'>"
                "Click 'Launch TensorBoard' to start monitoring.</p>"
            )
        )

        launch_btn.click(
            fn=launch_and_embed,
            inputs=[],
            outputs=tb_iframe,
            queue=False,
            api_visibility="private",
        )

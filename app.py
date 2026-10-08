# Make sure the config file exists
import os
import shutil
import sys
import subprocess


def pull_updates():
    directory = os.path.dirname(os.path.abspath(__file__))
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
        raise SystemExit("Resolve the existing Git conflicts before starting the WebUI.")
    if git("branch", "--show-current").stdout.strip() != "experimental":
        print("Automatic updates apply to experimental checkouts only; starting the current branch.", flush=True)
        return False
    previous = git("rev-parse", "HEAD").stdout.strip()
    print("Checking origin/experimental for updates...", flush=True)
    result = git("pull", "--ff-only", "--autostash", "origin", "experimental", check=False)
    message = (result.stdout + result.stderr).strip()
    if message:
        print(message, flush=True)
    if git("diff", "--name-only", "--diff-filter=U").stdout.strip():
        raise SystemExit("Git could not reapply local edits cleanly. Resolve the conflicts before starting the WebUI.")
    if result.returncode:
        raise RuntimeError(f"Experimental update failed with Git exit code {result.returncode}.")
    return git("rev-parse", "HEAD").stdout.strip() != previous


if __name__ == "__main__":
    _app_path = os.path.abspath(__file__)
    os.chdir(os.path.dirname(_app_path))
    if os.environ.pop("REDPANDA_EXPERIMENTAL_UPDATE_RESTART", "") != _app_path:
        try:
            if pull_updates():
                os.environ["REDPANDA_EXPERIMENTAL_UPDATE_RESTART"] = _app_path
                os.execv(sys.executable, [sys.executable, _app_path, *sys.argv[1:]])
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            print(f"Automatic update skipped: {error}. Starting the installed version.", file=sys.stderr, flush=True)


# We need the CWD for finding the config file, but while we're at it, add it to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)

# TODO: This path is regenerated all over the place in Applio
# should probably be in a static module for everything to reference
CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")

# The base config file to start from
CONFIG_TEMPLATE_PATH = os.path.join(now_dir, "assets", "config_template.json")

if not os.path.exists(CONFIG_PATH):
    print("Config file not found. Creating fresh from template.")
    shutil.copy(CONFIG_TEMPLATE_PATH, CONFIG_PATH)

# Plataform config
from rvc.lib.platform import platform_config

platform_config()

import argparse
import logging

import gradio as gr

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

_ARG_PARSER = argparse.ArgumentParser(
    description="Applio Web UI",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_ARG_PARSER.add_argument(
    "--port", type=int, default=DEFAULT_PORT, help="Server port (default: %(default)s)"
)
_ARG_PARSER.add_argument(
    "--server-name",
    type=str,
    default=DEFAULT_SERVER_NAME,
    help="Server hostname (default: %(default)s)",
)
_ARG_PARSER.add_argument(
    "--share", action="store_true", help="Create a public Gradio share link"
)
_ARG_PARSER.add_argument(
    "--open", action="store_true", help="Open the browser automatically"
)
_args, _ = _ARG_PARSER.parse_known_args()
_has_share = _args.share
_has_open = _args.open

# Set up logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress ConnectionResetError on Windows when a remote peer forcibly closes the
# connection during asyncio shutdown (WinError 10054 / ProactorBasePipeTransport).
if sys.platform == "win32":
    import asyncio.proactor_events as _pe

    _orig_ccl = _pe._ProactorBasePipeTransport._call_connection_lost

    def _ccl_patched(self, exc):
        try:
            _orig_ccl(self, exc)
        except ConnectionResetError:
            pass

    _pe._ProactorBasePipeTransport._call_connection_lost = _ccl_patched

# Fix Gradio NoneType error when entering an invalid value
gr.Number.preprocess = lambda self, payload: (
    None
    if payload is None
    or (self.minimum is not None and payload < self.minimum)
    or (self.maximum is not None and payload > self.maximum)
    else self.round_to_precision(payload, self.precision)
)

# detect gradio
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

# Zluda hijack
import rvc.lib.zluda

# Import Tabs
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.train.one_click_train import one_click_train_tab
from tabs.download.download import download_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.checkpoint_exporter.checkpoint_exporter import checkpoint_exporter_tab
from tabs.settings.settings import settings_tab
from tabs.tensorboard.tensorboard import tensorboard_tab

# Run prerequisites
from core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=True,
    beatrice=True,
)

# Initialize i18n
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Check installation
import assets.installation_checker as installation_checker

installation_checker.check_installation()

# Load theme
import assets.themes.loadThemes as loadThemes

my_applio = loadThemes.load_theme()

APP_CSS = """
footer { display: none !important; }
body {
    background: #181114 !important;
}
.gradio-container {
    width: 100% !important;
    max-width: 1420px !important;
    margin: 0 auto !important;
    padding: 0 28px 44px !important;
    box-sizing: border-box;
    background: transparent !important;
}
gradio-app > .gradio-container {
    width: 100% !important;
    max-width: 1420px !important;
    margin: 0 auto !important;
}
#applio-header {
    position: relative;
    overflow: hidden;
    align-items: center;
    min-height: 250px;
    margin: 26px 0 20px;
    padding: 28px 38px 24px 42px;
    border: 1px solid rgba(203, 161, 116, 0.28);
    border-radius: 20px;
    background: #291a1e;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
}
#applio-header::before {
    display: none;
}
#applio-intro {
    z-index: 1;
    gap: 10px;
}
#applio-title h2 {
    margin: 0 !important;
    color: #fff3df !important;
    font-size: clamp(2.4rem, 5vw, 4rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.06em;
}
#applio-title h2::after {
    content: "";
    display: block;
    width: 48px;
    height: 3px;
    margin-top: 12px;
    border-radius: 2px;
    background: #c96f45;
}
#applio-description {
    max-width: 620px;
}
#applio-description p {
    margin: 0 !important;
    color: #eadcca !important;
    font-size: 1.03rem !important;
    line-height: 1.65 !important;
}
#applio-github p {
    margin: 4px 0 0 !important;
}
#applio-github a {
    display: inline-flex;
    align-items: center;
    padding: 8px 13px;
    border: 1px solid rgba(199, 134, 134, 0.36);
    border-radius: 8px;
    color: #e0aaa2 !important;
    background: rgba(31, 18, 21, 0.62);
    text-decoration: none !important;
}
#applio-github a:hover {
    border-color: rgba(232, 184, 173, 0.72);
    background: rgba(199, 134, 134, 0.12);
}
#applio-mascot {
    z-index: 1;
    align-self: stretch;
    min-width: 190px !important;
    max-width: 260px !important;
    margin: -12px 12px -16px 0;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
#applio-mascot > div,
#applio-mascot .image-container {
    height: 100% !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
#applio-mascot img {
    width: 100% !important;
    height: 250px !important;
    object-fit: contain !important;
    filter: drop-shadow(0 10px 12px rgba(0, 0, 0, 0.28));
}
#main-tabs > .tab-nav {
    gap: 7px;
    padding: 7px;
    border: 1px solid rgba(196, 174, 150, 0.14);
    border-radius: 12px;
    background: rgba(34, 23, 27, 0.94);
}
#main-tabs > .tab-nav button {
    border: 0 !important;
    border-radius: 8px !important;
    color: #e0d1c0 !important;
    font-weight: 600 !important;
}
#main-tabs > .tab-nav button:hover {
    color: #f7ead7 !important;
    background: rgba(199, 134, 134, 0.1) !important;
}
#main-tabs > .tab-nav button.selected {
    color: #ffffff !important;
    background: #8f4650 !important;
}
#main-tabs > .tabitem {
    padding-top: 24px;
}
#main-tabs .gr-panel,
#main-tabs .gr-form {
    border-color: #66434a !important;
    background: #291b1f !important;
}
#main-tabs .gr-box {
    background: transparent !important;
    border-color: #5f3d44 !important;
}
#main-tabs .block-label,
#main-tabs .block-title {
    border-color: rgba(201, 121, 77, 0.4) !important;
    border-radius: 6px !important;
    color: #f7ead7 !important;
    background: #3d282e !important;
}
#main-tabs input,
#main-tabs textarea,
#main-tabs select {
    color: #fff4e8 !important;
    background: #211619 !important;
    border-color: #76505a !important;
}
#main-tabs input:focus,
#main-tabs textarea:focus,
#main-tabs select:focus {
    border-color: #c9794d !important;
    box-shadow: none !important;
}
#main-tabs .gr-accordion {
    border-color: #66434a !important;
    background: #291b1f !important;
}
#main-tabs .gr-accordion > .label-wrap {
    color: #f1e6d6 !important;
}
#main-tabs .upload-container {
    border-color: rgba(230, 161, 109, 0.68) !important;
    background: #291b1f !important;
}
#main-tabs .upload-container,
#main-tabs .upload-container *,
#main-tabs .gr-file-upload,
#main-tabs .gr-file-upload * {
    color: #fff4e8 !important;
}
#main-tabs .upload-container svg,
#main-tabs .gr-file-upload svg {
    color: #e8b86d !important;
    stroke: #e8b86d !important;
}
#main-tabs .upload-container .block-label,
#main-tabs .upload-container .block-label *,
#main-tabs .gr-file-upload .block-label,
#main-tabs .gr-file-upload .block-label * {
    color: #f7ead7 !important;
}
#main-tabs .upload-container .wrap,
#main-tabs .gr-file-upload .wrap {
    background: transparent !important;
}
#main-tabs .upload-container .or,
#main-tabs .gr-file-upload .or {
    color: #e0d1c0 !important;
}
#main-tabs input[type="range"] {
    accent-color: #c9794d;
}
#main-tabs input[type="checkbox"],
#main-tabs input[type="radio"] {
    width: 18px !important;
    height: 18px !important;
    flex: 0 0 18px !important;
    cursor: pointer;
    background-color: #211619 !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    background-size: 14px 14px !important;
    border: 1px solid #9a626b !important;
}
#main-tabs input[type="checkbox"] {
    border-radius: 4px !important;
}
#main-tabs input[type="checkbox"]:checked {
    background-color: #c96f45 !important;
    border-color: #e6a16d !important;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 16 16' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M3.5 8.4 6.6 11.3 12.6 4.8' fill='none' stroke='%23fff8ef' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") !important;
}
#main-tabs input[type="radio"] {
    border-radius: 50% !important;
}
#main-tabs input[type="radio"]:checked {
    background-color: #c96f45 !important;
    border-color: #e6a16d !important;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 16 16' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='8' cy='8' r='3.25' fill='%23fff8ef'/%3E%3C/svg%3E") !important;
}
#main-tabs input[type="checkbox"]:focus-visible,
#main-tabs input[type="radio"]:focus-visible {
    outline: 2px solid #e6a16d !important;
    outline-offset: 2px;
}
#main-tabs label:has(input[type="checkbox"]:checked) .label-text,
#main-tabs label.selected span {
    color: #fff3df !important;
    font-weight: 600 !important;
}
#main-tabs input[type="checkbox"]:disabled,
#main-tabs input[type="radio"]:disabled {
    cursor: not-allowed;
    opacity: 0.5;
}
#main-tabs .tabitem .tab-nav button {
    color: #e0d1c0 !important;
}
#main-tabs .tabitem .tab-nav button:hover {
    color: #f7ead7 !important;
}
#main-tabs .tabitem .tab-nav button.selected {
    color: #e0aaa2 !important;
    border-bottom-color: #c78686 !important;
}
#normalization-db [data-normalization-off="true"],
#normalization-db-batch [data-normalization-off="true"] {
    position: relative;
}
#normalization-db [data-normalization-off="true"] [data-testid="number-input"],
#normalization-db-batch [data-normalization-off="true"] [data-testid="number-input"] {
    color: transparent !important;
    caret-color: transparent !important;
}
#normalization-db [data-normalization-off="true"]::before,
#normalization-db-batch [data-normalization-off="true"]::before {
    content: "Off";
    position: absolute;
    z-index: 1;
    top: 50%;
    left: 9px;
    color: #fff3df;
    font-size: 0.875rem;
    pointer-events: none;
    transform: translateY(-50%);
}
@media (max-width: 720px) {
    .gradio-container { padding: 0 14px 28px !important; }
    #applio-header { min-height: 0; padding: 28px 24px 0; }
    #applio-mascot { display: none; }
    #main-tabs > .tab-nav { overflow-x: auto; }
}
"""

APP_JS = """
let favicon = document.querySelector("link[rel~='icon']");
if (!favicon) {
    favicon = document.createElement("link");
    favicon.rel = "icon";
    document.head.appendChild(favicon);
}
favicon.type = "image/x-icon";
favicon.href = "favicon.ico?v=red-panda-ico-1";

const updateNormalizationLabels = () => {
    document.querySelectorAll("#normalization-db, #normalization-db-batch").forEach((root) => {
        const label = root.querySelector("[data-testid='min-value']");
        const range = root.querySelector("[data-testid='range-input']");
        const number = root.querySelector("[data-testid='number-input']");
        const container = number?.parentElement;
        if (label && label.textContent !== "Off") label.textContent = "Off";
        if (!range || !number || !container) return;
        const sync = () => {
            const off = Number(range.value) === Number(range.min);
            container.dataset.normalizationOff = String(off);
            if (off) range.setAttribute("aria-valuetext", "Off");
            else range.removeAttribute("aria-valuetext");
        };
        sync();
        if (range.dataset.normalizationBound !== "true") {
            range.dataset.normalizationBound = "true";
            range.addEventListener("input", () => requestAnimationFrame(sync));
            number.addEventListener("input", () => requestAnimationFrame(sync));
        }
    });
};
updateNormalizationLabels();
new MutationObserver(updateNormalizationLabels).observe(document.body, {
    childList: true,
    subtree: true
});
"""

# Define Gradio interface
with gr.Blocks(
    title="Redpanda-rvc",
    **(
        {
            "theme": my_applio,
            "css": APP_CSS,
        }
        if not GRADIO_6
        else {}
    ),
) as Applio:
    with gr.Row(elem_id="applio-header"):
        with gr.Column(elem_id="applio-intro", scale=5):
            gr.Markdown("## Redpanda-rvc", elem_id="applio-title")
            gr.Markdown(
                i18n(
                    "A simple, high-quality voice conversion tool focused on ease of use and performance."
                ),
                elem_id="applio-description",
            )
            gr.Markdown(
                i18n("[GitHub](https://github.com/redpanda343/redpanda-rvc)"),
                elem_id="applio-github",
            )
        gr.Image(
            value="assets/applio_mascot.png",
            show_label=False,
            buttons=[],
            interactive=False,
            container=False,
            height=250,
            elem_id="applio-mascot",
            scale=2,
        )

    with gr.Tabs(elem_id="main-tabs"):
        with gr.Tab(i18n("Inference")):
            inference_tab()

        with gr.Tab(i18n("Training")):
            train_tab()

        with gr.Tab(i18n("One-click Training")):
            one_click_train_tab()

        with gr.Tab(i18n("Voice Blender")):
            voice_blender_tab()

        with gr.Tab(i18n("Checkpoint Exporter")):
            checkpoint_exporter_tab()

        with gr.Tab(i18n("Download")):
            download_tab()

        with gr.Tab(i18n("Settings")):
            settings_tab()

        with gr.Tab(i18n("TensorBoard")):
            tensorboard_tab()



def launch_gradio(server_name: str, server_port: int, auth=None) -> None:
    if auth is None:
        from rvc.lib.tools.web_auth import build_launch_auth

        auth = build_launch_auth(server_name, _has_share)
    app, local_url, share_url = Applio.launch(
        favicon_path="assets/red_panda_favicon.ico",
        share=_has_share,
        inbrowser=False,
        server_name=server_name,
        server_port=server_port,
        js=APP_JS,
        auth=auth,
        auth_message="Sign in to access this Redpanda-rvc instance.",
        prevent_thread_lock=True,
        **(
            {
                "theme": my_applio,
                "css": APP_CSS,
            }
            if GRADIO_6
            else {}
        ),
    )
    from rvc.lib.tools.tensorboard_proxy import register_tensorboard_proxy

    try:
        register_tensorboard_proxy(app)
        if _has_open:
            import webbrowser

            webbrowser.open(share_url or local_url)
        Applio.block_thread()
    except Exception:
        Applio.close()
        raise


if __name__ == "__main__":
    from rvc.lib.tools.web_auth import build_launch_auth

    port = _args.port
    server = _args.server_name
    launch_credentials = build_launch_auth(server, _has_share)

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(server, port, launch_credentials)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break

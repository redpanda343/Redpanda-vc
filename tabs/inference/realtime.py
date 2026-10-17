import os

import gradio as gr

from assets.i18n.i18n import I18nAuto
from shared.tools import realtime_server

i18n = I18nAuto()
now_dir = os.getcwd()

REALTIME_CSS = """
.rp-rt {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 14px 16px;
    border: 1px solid rgba(203, 161, 116, 0.28);
    border-radius: 10px;
    background: #291b1f;
}
.rp-rt-devices {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: flex-end;
}
.rp-rt-devices label {
    display: flex;
    flex: 1 1 220px;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
    color: #e0d1c0;
    font-size: 0.85rem;
}
.rp-rt-devices select {
    padding: 7px 8px;
    border: 1px solid #76505a;
    border-radius: 6px;
    color: #fff4e8;
    background: #211619;
}
.rp-rt-devices button {
    padding: 8px 12px;
    border: 1px solid #9a626b;
    border-radius: 6px;
    color: #f7ead7;
    background: #3d282e;
    cursor: pointer;
}
.rp-rt-status {
    color: #fff3df;
    font-weight: 600;
}
.rp-rt-stats {
    color: #c9b8a6;
    font-size: 0.85rem;
    font-variant-numeric: tabular-nums;
}
"""

REALTIME_JS = r"""
const CHUNK_SECONDS = 0.04;
const MAX_IN_FLIGHT = 2;
const MAX_MERGE_SECONDS = 0.5;
const MAX_QUEUED_SECONDS = 1.0;
const STORAGE_PREFIX = "redpanda-realtime-";
const WORKLET = `
class RedpandaRealtime extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const settings = options.processorOptions;
        this.chunk = settings.chunk;
        this.capture = new Float32Array(this.chunk);
        this.filled = 0;
        this.ring = new Float32Array(settings.capacity);
        this.read = 0;
        this.count = 0;
        this.target = settings.target;
        this.limit = settings.limit;
        this.playing = false;
        this.underruns = 0;
        this.trimmed = 0;
        this.report = Math.round(sampleRate / 4);
        this.since = 0;
        this.port.onmessage = (event) => {
            const data = event.data;
            if (data.type === "audio") {
                this.push(data.samples);
            } else if (data.type === "buffer") {
                this.target = data.target;
                this.limit = data.limit;
            }
        };
    }

    push(samples) {
        const capacity = this.ring.length;
        if (samples.length > capacity) {
            samples = samples.subarray(samples.length - capacity);
        }
        if (this.count + samples.length > capacity) {
            this.drop(this.count + samples.length - capacity);
        }
        const write = (this.read + this.count) % capacity;
        const first = Math.min(samples.length, capacity - write);
        this.ring.set(samples.subarray(0, first), write);
        this.ring.set(samples.subarray(first), 0);
        this.count += samples.length;
        if (this.count > this.limit) {
            this.trimmed += 1;
            this.drop(this.count - this.target);
        }
    }

    drop(amount) {
        this.read = (this.read + amount) % this.ring.length;
        this.count -= amount;
    }

    process(inputs, outputs) {
        const input = inputs[0] && inputs[0][0];
        if (input) {
            let offset = 0;
            while (offset < input.length) {
                const take = Math.min(input.length - offset, this.chunk - this.filled);
                this.capture.set(input.subarray(offset, offset + take), this.filled);
                this.filled += take;
                offset += take;
                if (this.filled === this.chunk) {
                    this.port.postMessage({ type: "capture", samples: this.capture }, [this.capture.buffer]);
                    this.capture = new Float32Array(this.chunk);
                    this.filled = 0;
                }
            }
        }
        const output = outputs[0];
        const frames = output[0].length;
        if (!this.playing && this.count >= this.target) {
            this.playing = true;
        }
        if (this.playing && this.count < frames) {
            this.playing = false;
            this.underruns += 1;
        }
        if (this.playing) {
            const capacity = this.ring.length;
            const channel = output[0];
            const first = Math.min(frames, capacity - this.read);
            channel.set(this.ring.subarray(this.read, this.read + first));
            channel.set(this.ring.subarray(0, frames - first), first);
            this.drop(frames);
            for (let index = 1; index < output.length; index++) {
                output[index].set(channel);
            }
        }
        this.since += frames;
        if (this.since >= this.report) {
            this.since = 0;
            this.port.postMessage({ type: "level", buffered: this.count, underruns: this.underruns, trimmed: this.trimmed });
        }
        return true;
    }
}
registerProcessor("redpanda-realtime", RedpandaRealtime);
`;

if (!window.redpandaRealtime) {
    const state = {
        root: null,
        context: null,
        module: null,
        workletUrl: "",
        node: null,
        source: null,
        stream: null,
        events: null,
        token: "",
        ready: false,
        queue: [],
        queued: 0,
        inFlight: 0,
        offset: 0,
        captured: 0,
        captures: [],
        dropped: 0,
        block: 0,
        chunk: 0,
        rate: 0,
        delay: 0,
        rtt: 0,
        buffered: 0,
        underruns: 0,
        server: null,
        buffer: 0.12,
        audioUrl: "",
    };
    const part = (name) => state.root && state.root.querySelector('[data-rt="' + name + '"]');
    const message = (error) => (error && error.message ? error.message : String(error));
    const setStatus = (text) => {
        const node = part("status");
        if (node) node.textContent = text;
    };
    const remember = (name, value) => {
        try {
            localStorage.setItem(STORAGE_PREFIX + name, value);
        } catch (error) {}
    };
    const recall = (name) => {
        try {
            return localStorage.getItem(STORAGE_PREFIX + name) || "";
        } catch (error) {
            return "";
        }
    };
    const endpoint = (path) => new URL(path, new URL(".", window.location.href));
    const canPickOutput = () => Boolean(window.AudioContext && AudioContext.prototype.setSinkId);

    const fill = (select, devices, saved) => {
        if (!select) return;
        const current = select.value || saved;
        const options = [new Option("Default", "")];
        devices.forEach((device, index) => {
            if (device.deviceId && device.deviceId !== "default" && device.deviceId !== "communications") {
                options.push(new Option(device.label || "Device " + (index + 1), device.deviceId));
            }
        });
        select.replaceChildren(...options);
        select.value = options.some((option) => option.value === current) ? current : "";
    };

    const refreshDevices = async (ask) => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
        try {
            let devices = await navigator.mediaDevices.enumerateDevices();
            if (ask && !devices.some((device) => device.kind === "audioinput" && device.label)) {
                const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
                probe.getTracks().forEach((track) => track.stop());
                devices = await navigator.mediaDevices.enumerateDevices();
            }
            fill(part("input"), devices.filter((device) => device.kind === "audioinput"), recall("input"));
            fill(part("output"), devices.filter((device) => device.kind === "audiooutput"), recall("output"));
        } catch (error) {
            setStatus("Could not list audio devices: " + message(error));
        }
    };

    const applySink = async () => {
        const select = part("output");
        if (!state.context || !canPickOutput() || !select) return;
        try {
            await state.context.setSinkId(select.value || "");
        } catch (error) {
            setStatus("Could not switch the speaker: " + message(error));
        }
    };

    const renderStats = () => {
        const node = part("stats");
        if (!node) return;
        if (!state.ready || !state.rate) {
            node.textContent = "";
            return;
        }
        const buffered = (1000 * state.buffered) / state.rate;
        const total = state.rtt + buffered + (1000 * (state.block + state.chunk)) / state.rate + 1000 * state.delay;
        const server = state.server || {};
        let text = "Delay about " + Math.round(total) + " ms   ·   Round trip " + Math.round(state.rtt) + " ms   ·   Buffered " + Math.round(buffered) + " ms   ·   Processing " + (server.blockMs || 0) + " ms   ·   Dropouts " + state.underruns;
        if (server.lost) text += "   ·   Lost chunks " + server.lost;
        if (state.dropped) text += "   ·   Dropped uploads " + state.dropped;
        if (server.skipped) text += "   ·   Skipped blocks " + server.skipped;
        node.textContent = text;
    };

    const bufferOptions = (seconds) => ({
        target: Math.max(128, Math.round(state.rate * seconds)),
        limit: Math.round(state.rate * (seconds + Math.max(seconds, 0.15))),
    });

    const flush = () => {
        const limit = Math.round(state.rate * MAX_MERGE_SECONDS);
        while (state.inFlight < MAX_IN_FLIGHT && state.queue.length) {
            const parts = [state.queue.shift()];
            let total = parts[0].length;
            while (state.queue.length && total + state.queue[0].length <= limit) {
                const part = state.queue.shift();
                parts.push(part);
                total += part.length;
            }
            state.queued -= total;
            const pcm = new Int16Array(total);
            let position = 0;
            for (const part of parts) {
                for (let index = 0; index < part.length; index++) {
                    const value = Math.max(-1, Math.min(1, part[index]));
                    pcm[position++] = value < 0 ? value * 32768 : value * 32767;
                }
            }
            const offset = state.offset;
            const token = state.token;
            state.offset += total;
            state.inFlight += 1;
            fetch(state.audioUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/octet-stream",
                    "X-Realtime-Token": token,
                    "X-Realtime-Offset": String(offset),
                },
                body: pcm.buffer,
                credentials: "same-origin",
                cache: "no-store",
            })
                .catch(() => {})
                .finally(() => {
                    if (state.token === token) {
                        state.inFlight -= 1;
                        flush();
                    }
                });
        }
    };

    const onCapture = (samples) => {
        state.captured += samples.length;
        state.captures.push([state.captured, performance.now()]);
        if (state.captures.length > 400) state.captures.shift();
        state.queue.push(samples);
        state.queued += samples.length;
        const limit = Math.round(state.rate * MAX_QUEUED_SECONDS);
        while (state.queued > limit && state.queue.length > 1) {
            const dropped = state.queue.shift();
            state.queued -= dropped.length;
            state.offset += dropped.length;
            state.dropped += 1;
        }
        flush();
    };

    const onWorklet = (event) => {
        const data = event.data;
        if (data.type === "capture") {
            if (state.ready) onCapture(data.samples);
        } else if (data.type === "level") {
            state.buffered = data.buffered;
            state.underruns = data.underruns;
            renderStats();
        }
    };

    const onAudio = (event) => {
        if (!state.node) return;
        const text = event.data;
        const space = text.indexOf(" ");
        const index = Number(text.slice(0, space));
        const binary = atob(text.slice(space + 1));
        const bytes = new Uint8Array(binary.length);
        for (let position = 0; position < binary.length; position++) {
            bytes[position] = binary.charCodeAt(position);
        }
        const pcm = new Int16Array(bytes.buffer);
        const samples = new Float32Array(pcm.length);
        for (let position = 0; position < pcm.length; position++) {
            samples[position] = pcm[position] / 32768;
        }
        state.node.port.postMessage({ type: "audio", samples }, [samples.buffer]);
        const sample = (index + 1) * state.block;
        while (state.captures.length && state.captures[0][0] < sample) state.captures.shift();
        if (state.captures.length) state.rtt = performance.now() - state.captures[0][1];
    };

    const stop = () => {
        const events = state.events;
        state.events = null;
        state.ready = false;
        state.token = "";
        if (events) events.close();
        if (state.source) state.source.disconnect();
        if (state.node) {
            state.node.port.onmessage = null;
            state.node.disconnect();
        }
        if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
        state.source = null;
        state.node = null;
        state.stream = null;
        renderStats();
        setStatus("Stopped");
    };

    const fail = (text) => {
        stop();
        setStatus(text);
    };

    const unlock = () => {
        try {
            if (!state.context || state.context.state === "closed") {
                state.context = new AudioContext({ latencyHint: "interactive" });
                state.module = null;
            }
            state.context.resume();
        } catch (error) {}
    };

    const setBuffer = (seconds) => {
        state.buffer = Number(seconds) || state.buffer;
        if (state.node && state.rate) {
            state.node.port.postMessage({ type: "buffer", ...bufferOptions(state.buffer) });
        }
    };

    const start = async (token, seconds) => {
        if (!token) return;
        stop();
        if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setStatus("The browser only allows the microphone on HTTPS pages or on localhost. Open the share link or http://127.0.0.1 instead.");
            return;
        }
        state.token = token;
        try {
            unlock();
            const context = state.context;
            if (!context) throw new Error("this browser has no Web Audio support.");
            setStatus("Opening the microphone...");
            const input = part("input") ? part("input").value : "";
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    deviceId: input ? { exact: input } : undefined,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    channelCount: 1,
                },
            });
            if (state.token !== token) {
                stream.getTracks().forEach((track) => track.stop());
                return;
            }
            state.stream = stream;
            refreshDevices(false);
            await applySink();
            if (state.module !== context) {
                if (!state.workletUrl) {
                    state.workletUrl = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
                }
                await context.audioWorklet.addModule(state.workletUrl);
                state.module = context;
            }
            state.rate = context.sampleRate;
            state.chunk = Math.round(state.rate * CHUNK_SECONDS);
            state.queue = [];
            state.queued = 0;
            state.inFlight = 0;
            state.offset = 0;
            state.captured = 0;
            state.captures = [];
            state.dropped = 0;
            state.rtt = 0;
            state.buffered = 0;
            state.underruns = 0;
            state.server = null;
            state.buffer = Number(seconds) || state.buffer;
            state.node = new AudioWorkletNode(context, "redpanda-realtime", {
                numberOfInputs: 1,
                numberOfOutputs: 1,
                outputChannelCount: [2],
                processorOptions: { chunk: state.chunk, capacity: Math.round(state.rate * 4), ...bufferOptions(state.buffer) },
            });
            state.node.port.onmessage = onWorklet;
            state.source = context.createMediaStreamSource(stream);
            state.source.connect(state.node);
            state.node.connect(context.destination);
            state.audioUrl = endpoint("realtime/audio").toString();
            const url = endpoint("realtime/stream");
            url.searchParams.set("token", token);
            url.searchParams.set("rate", String(state.rate));
            const events = new EventSource(url.toString());
            state.events = events;
            events.addEventListener("status", (event) => setStatus(JSON.parse(event.data).message));
            events.addEventListener("ready", (event) => {
                const info = JSON.parse(event.data);
                state.block = info.block;
                state.delay = info.delay;
                state.ready = true;
                setStatus("Running at " + state.rate / 1000 + " kHz. Use headphones so the converted voice does not reach the microphone.");
            });
            events.addEventListener("audio", onAudio);
            events.addEventListener("stats", (event) => {
                state.server = JSON.parse(event.data);
                renderStats();
            });
            events.addEventListener("failure", (event) => fail("Stopped: " + JSON.parse(event.data).message));
            events.onerror = () => {
                if (state.events === events) {
                    fail(state.ready ? "Stopped: the connection to the server was lost." : "Could not start real-time conversion.");
                }
            };
        } catch (error) {
            fail("Could not start: " + message(error));
        }
    };

    const attach = (root) => {
        state.root = root;
        root.innerHTML =
            '<div class="rp-rt">' +
            '<div class="rp-rt-devices">' +
            '<label>Microphone<select data-rt="input"><option value="">Default</option></select></label>' +
            '<label>Speaker<select data-rt="output"><option value="">Default</option></select></label>' +
            '<button type="button" data-rt="refresh">Refresh devices</button>' +
            "</div>" +
            '<div class="rp-rt-status" data-rt="status">Stopped</div>' +
            '<div class="rp-rt-stats" data-rt="stats"></div>' +
            "</div>";
        part("refresh").addEventListener("click", () => refreshDevices(true));
        ["input", "output"].forEach((name) => {
            part(name).addEventListener("change", () => {
                remember(name, part(name).value);
                if (name === "output") applySink();
            });
        });
        if (!canPickOutput()) {
            part("output").disabled = true;
            part("output").title = "This browser always plays through its default output.";
        }
        refreshDevices(false);
        if (state.ready) setStatus("Running");
    };

    window.redpandaRealtime = { attach, unlock, start, stop, setBuffer };
}
window.redpandaRealtime.attach(element);
"""


def _user(request):
    return (request.username if request is not None else None) or "local"


def start_realtime(
    model,
    index,
    vocoder,
    pitch,
    index_rate,
    f0_method,
    embedder_model,
    speaker_id,
    threshold,
    rms_mix_rate,
    input_gain_db,
    output_gain_db,
    input_noise_reduce,
    output_noise_reduce,
    block_time,
    crossfade_time,
    extra_time,
    lookahead_time,
    seamless,
    cached_context,
    request: gr.Request,
):
    if not model:
        raise gr.Error(i18n("Select a voice model first."))
    settings = {
        "model_path": os.path.abspath(os.path.join(now_dir, model)),
        "index_path": os.path.abspath(os.path.join(now_dir, index)) if index else "",
        "rectified_vocoder_path": vocoder or "",
        "rectified_steps": 0,
        "rectified_flow_window": True,
        "embedder_model": embedder_model,
        "index_rate": float(index_rate),
        "pitch": int(pitch),
        "speaker_id": int(speaker_id or 0),
        "formant_shift": 0.0,
        "vq_neighbors": 0,
        "f0_method": f0_method,
        "block_time": float(block_time),
        "crossfade_time": float(crossfade_time),
        "extra_time": float(extra_time),
        "lookahead_time": float(lookahead_time),
        "seamless": bool(seamless),
        "cached_context": bool(cached_context),
        "safety_buffer": 0.0,
        "threshold": int(threshold),
        "rms_mix_rate": float(rms_mix_rate),
        "input_gain_db": float(input_gain_db),
        "output_gain_db": float(output_gain_db),
        "input_noise_reduce": bool(input_noise_reduce),
        "output_noise_reduce": bool(output_noise_reduce),
        "passthrough": False,
    }
    try:
        return realtime_server.create_session(_user(request), settings)
    except Exception as error:
        raise gr.Error(str(error))


def stop_realtime(token, request: gr.Request):
    if token:
        realtime_server.close_session(token, _user(request))
    return ""


def update_realtime(
    token,
    pitch,
    index_rate,
    f0_method,
    threshold,
    rms_mix_rate,
    input_gain_db,
    output_gain_db,
    input_noise_reduce,
    output_noise_reduce,
    request: gr.Request,
):
    if not token:
        return
    realtime_server.update_session(
        token,
        _user(request),
        pitch=int(pitch),
        index_rate=float(index_rate),
        f0_method=f0_method,
        threshold=int(threshold),
        rms_mix_rate=float(rms_mix_rate),
        input_gain_db=float(input_gain_db),
        output_gain_db=float(output_gain_db),
        input_noise_reduce=bool(input_noise_reduce),
        output_noise_reduce=bool(output_noise_reduce),
    )


def realtime_tab(model_file, index_file, rectified_vocoder_path):
    gr.Markdown(
        i18n(
            "Converts your microphone in real time through this browser, also over a public share link. The browser only allows the microphone on HTTPS pages or on localhost, so open the share link or http://127.0.0.1. Use headphones so the converted voice does not reach the microphone."
        )
    )
    gr.HTML(value="", js_on_load=REALTIME_JS, css_template=REALTIME_CSS, elem_id="realtime-panel")
    with gr.Row():
        start_button = gr.Button(i18n("Start"), variant="primary")
        stop_button = gr.Button(i18n("Stop"))
    token = gr.Textbox(visible=False)
    with gr.Row():
        pitch = gr.Slider(
            minimum=-24,
            maximum=24,
            step=1,
            value=0,
            label=i18n("Pitch"),
            info=i18n("Shift in semitones. +12 is one octave up."),
            interactive=True,
        )
        index_rate = gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            value=0,
            label=i18n("Index Rate"),
            info=i18n("How strongly the feature index pulls the voice toward the training data."),
            interactive=True,
        )
    with gr.Row():
        f0_method = gr.Radio(
            label=i18n("Pitch extraction algorithm"),
            choices=["rmvpe", "swift", "beatrice", ("parselmouth", "pm")],
            value="rmvpe",
            interactive=True,
        )
        embedder_model = gr.Radio(
            label=i18n("Embedder Model"),
            info=i18n("Must match the embedder the model was trained with."),
            choices=["contentvec", "spin-v2"],
            value="contentvec",
            interactive=True,
        )
        speaker_id = gr.Number(
            label=i18n("Speaker ID"),
            value=0,
            precision=0,
            minimum=0,
            interactive=True,
        )
    with gr.Accordion(i18n("Audio"), open=False):
        with gr.Row():
            threshold = gr.Slider(
                minimum=-60,
                maximum=0,
                step=1,
                value=-60,
                label=i18n("Noise Gate"),
                info=i18n("Input quieter than this (dB) is muted. -60 turns the gate off."),
                interactive=True,
            )
            rms_mix_rate = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.01,
                value=1,
                label=i18n("RMS Mix"),
                info=i18n("0 follows your input loudness, 1 keeps the model's own loudness."),
                interactive=True,
            )
        with gr.Row():
            input_gain_db = gr.Slider(
                minimum=-24, maximum=24, step=0.5, value=0, label=i18n("Input Gain (dB)"), interactive=True
            )
            output_gain_db = gr.Slider(
                minimum=-24, maximum=24, step=0.5, value=0, label=i18n("Output Gain (dB)"), interactive=True
            )
        with gr.Row():
            input_noise_reduce = gr.Checkbox(label=i18n("Input Noise Reduction"), value=False, interactive=True)
            output_noise_reduce = gr.Checkbox(label=i18n("Output Noise Reduction"), value=False, interactive=True)
    with gr.Accordion(i18n("Performance"), open=False):
        with gr.Row():
            block_time = gr.Slider(
                minimum=0.05,
                maximum=1.5,
                step=0.01,
                value=0.2,
                label=i18n("Block Time"),
                info=i18n("Seconds of audio per step. Lower is less delay but more GPU load and more network requests."),
                interactive=True,
            )
            buffer_time = gr.Slider(
                minimum=0.02,
                maximum=1.0,
                step=0.01,
                value=0.12,
                label=i18n("Playback Buffer"),
                info=i18n("Seconds of converted audio the browser keeps ready to cover network delays. Raise it if Dropouts keeps growing. Applies while running."),
                interactive=True,
            )
        with gr.Row():
            crossfade_time = gr.Slider(
                minimum=0.01,
                maximum=0.1,
                step=0.01,
                value=0.04,
                label=i18n("Crossfade"),
                info=i18n("Seconds each block blends into the next. Capped at the block time."),
                interactive=True,
            )
            extra_time = gr.Slider(
                minimum=0.05,
                maximum=5.0,
                step=0.01,
                value=2.0,
                label=i18n("Extra Context"),
                info=i18n("Seconds of past audio the model sees. More is smoother but slower."),
                interactive=True,
            )
            lookahead_time = gr.Slider(
                minimum=0.0,
                maximum=0.3,
                step=0.01,
                value=0.0,
                label=i18n("Lookahead"),
                info=i18n("Seconds of future audio the model sees past each block, added to the delay."),
                interactive=True,
            )
        with gr.Row():
            seamless = gr.Checkbox(
                label=i18n("Seamless Blocks"),
                info=i18n("Makes consecutive blocks line up exactly. Uses more GPU time. RVC models only."),
                value=False,
                interactive=True,
            )
            cached_context = gr.Checkbox(
                label=i18n("Cached Context"),
                info=i18n("Reuses ContentVec results for older context, so a long extra context stays cheap. Needs about 1 s of extra context or more. RVC models only."),
                value=True,
                interactive=True,
            )
    gr.Markdown(
        i18n(
            "Block Time, Crossfade, Extra Context, Lookahead, the embedder, the speaker and the toggles above them apply on the next Start. Pitch, Index Rate, the pitch algorithm and the Audio settings apply while running."
        )
    )

    start_button.click(
        fn=start_realtime,
        inputs=[
            model_file,
            index_file,
            rectified_vocoder_path,
            pitch,
            index_rate,
            f0_method,
            embedder_model,
            speaker_id,
            threshold,
            rms_mix_rate,
            input_gain_db,
            output_gain_db,
            input_noise_reduce,
            output_noise_reduce,
            block_time,
            crossfade_time,
            extra_time,
            lookahead_time,
            seamless,
            cached_context,
        ],
        outputs=[token],
        js="(...args) => { window.redpandaRealtime && window.redpandaRealtime.unlock(); return args; }",
    ).success(
        fn=None,
        inputs=[token, buffer_time],
        outputs=None,
        js="(token, buffer) => { window.redpandaRealtime && window.redpandaRealtime.start(token, buffer); }",
    )
    stop_button.click(
        fn=stop_realtime,
        inputs=[token],
        outputs=[token],
        js="(token) => { window.redpandaRealtime && window.redpandaRealtime.stop(); return [token]; }",
    )
    live = [
        pitch,
        index_rate,
        f0_method,
        threshold,
        rms_mix_rate,
        input_gain_db,
        output_gain_db,
        input_noise_reduce,
        output_noise_reduce,
    ]
    for component in live:
        component.change(fn=update_realtime, inputs=[token, *live], outputs=None, show_progress="hidden")
    buffer_time.change(
        fn=None,
        inputs=[buffer_time],
        outputs=None,
        js="(buffer) => { window.redpandaRealtime && window.redpandaRealtime.setBuffer(buffer); }",
    )

import gradio as gr
import torch
from TTS.api import TTS
import os
from torch.serialization import safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from pydub import AudioSegment

os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load XTTS safely in PyTorch >=2.6
with safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def clone(text, audio, language):
    if audio is None or not os.path.exists(audio):
        return "Please upload a valid voice sample."

    if audio.endswith(".mp3"):
        converted = audio.replace(".mp3", ".wav")
        AudioSegment.from_mp3(audio).export(converted, format="wav")
        audio = converted

    out_path = "./output.wav"
    tts.tts_to_file(text=text, speaker_wav=audio, language=language, file_path=out_path)

    return out_path

examples = [
    ["Hey! It's me Dorthy, from the Wizard of Oz. Type in whatever you'd like me to say.", "./audio/Wizard-of-Oz-Dorthy.wav", "en"],
    ["It's me Vito Corleone, from the Godfather. Type in whatever you'd like me to say.", "./audio/Godfather.wav", "en"],
    ["Hey, it's me Paris Hilton. Type in whatever you'd like me to say.", "./audio/Paris-Hilton.mp3", "en"],
    ["Hey, it's me Megan Fox from Transformers. Type in whatever you'd like me to say.", "./audio/Megan-Fox.mp3", "en"],
    ["Hey there, it's me Jeff Goldblum. Type in whatever you'd like me to say.", "./audio/Jeff-Goldblum.mp3", "en"],
    ["Hey there, it's me Heath Ledger as the Joker. Type in whatever you'd like me to say.", "./audio/Heath-Ledger.mp3", "en"]
]

iface = gr.Interface(
    fn=clone,
    inputs=[
        gr.Textbox(label='Text'),
        gr.Audio(type='filepath', label='Voice reference audio file'),
        gr.Dropdown(
            choices=["en", "fr", "es", "de", "it", "pl", "ar", "zh", "ru", "ja"],
            value="en",
            label="Select Language"
        )
    ],
    outputs=gr.Audio(type='filepath'),
    title='Voice Clone',
    description="""
    by 3VO | Youniss Jaafil
    """,
    theme=gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate"),
    examples=examples
)

iface.launch(share=True, debug=True)

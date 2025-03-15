import random
import sys
from importlib.resources import files

import soundfile as sf
import tqdm
from cached_path import cached_path
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    transcribe,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from f5_tts.model import DiT, UNetT  # noqa: F401. used for config
from f5_tts.model.utils import seed_everything


class F5TTS:
    def __init__(
        self,
        model="F5TTS_v1_Base",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_local_path=None,
        device=None,
        hf_cache_dir=None,
    ):
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
        model_cls = globals()[model_cfg.model.backbone]
        model_arc = model_cfg.model.arch

        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

        self.ode_method = ode_method
        self.use_ema = use_ema

        if device is not None:
            self.device = device
        else:
            import torch

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        # Load models
        self.vocoder = load_vocoder(
            self.mel_spec_type, vocoder_local_path is not None, vocoder_local_path, self.device, hf_cache_dir
        )

        repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

        # override for previous models
        if model == "F5TTS_Base":
            if self.mel_spec_type == "vocos":
                ckpt_step = 1200000
            elif self.mel_spec_type == "bigvgan":
                model = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif model == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1200000
        if model == "F5TTS_v1_Base":
            ckpt_step = 1250000
        else:
            raise ValueError(f"Unknown model type: {model}")

        if not ckpt_file:
            hf = f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"
            print(f"ckpt_file: {hf}")
            ckpt_file = str(cached_path(hf, cache_dir=hf_cache_dir))

        self.ema_model = load_model(
            model_cls, model_arc, ckpt_file, self.mel_spec_type, vocab_file, self.ode_method, self.use_ema, self.device
        )

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spec, file_spec):
        save_spectrogram(spec, file_spec)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spec=None,
        seed=None,
    ):
        if seed is None:
            self.seed = random.randint(0, sys.maxsize)
        seed_everything(self.seed)

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, device=self.device)

        wav, sr, spec = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spec is not None:
            self.export_spectrogram(spec, file_spec)

        return wav, sr, spec



def infer(text: str, f5tts: F5TTS, speed: float = 1.0):
    wav, sr, spec = f5tts.infer(
        ref_file=str(files("f5_tts").joinpath("infer/examples/basic/eng.wav")),
        ref_text="",
        gen_text=text,
        seed=None,
        speed=speed,
    )
    return wav, sr, spec

import io
def export_wav(wav, sr, sample_rate):
    file_stream = io.BytesIO()
    sf.write(file_stream, wav, sample_rate, format="WAV")
    file_stream.seek(0)
    return file_stream

import torch
import torchaudio
def get_ref_audio_len():
    ref_file=str(files("f5_tts").joinpath("infer/examples/basic/eng.wav"))
    (audio, sr) = torchaudio.load(ref_file, format="wav")
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < 0.1:
        audio = audio * 0.1 / rms
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        audio = resampler(audio)
    audio = audio.to("cuda")
    hop_length = 256
    token_len = audio.shape[-1] // hop_length
    print(f"ref_token_len: {token_len}")
    return token_len

from pydantic import BaseModel
import fastapi
import uvicorn
from fastapi.responses import StreamingResponse, JSONResponse
import time

class TTSRequest(BaseModel):
    text: str
    speed: float

app = fastapi.FastAPI()
f5tts = F5TTS()

@app.post("/zero_shot")
def zero_shot(request: TTSRequest):
    text = request.text
    start_time = time.time()
    wav, sr, spec = infer(text, f5tts, request.speed)
    end_time = time.time()
    print(f"inference time: {end_time - start_time}")
    return StreamingResponse(export_wav(wav, sr, f5tts.target_sample_rate), media_type="audio/wav")

@app.get("/ref_token_len")
def ref_audio_len():
    token_len = get_ref_token_len()
    return JSONResponse({"ref_audio_len": token_len})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=38100)

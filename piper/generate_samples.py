#!/usr/bin/env python3
import itertools as it
import json
import logging
import os
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import librosa
import webrtcvad
from piper_phonemize import phonemize_espeak
from tqdm.autonotebook import trange
from . import PiperVoice

_DIR = Path(__file__).parent.parent
_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


# Main generation function
def generate_samples(
    text: Union[List[str], str],
    output_dir: Union[str, Path],
    max_samples: Optional[int] = None,
    file_names: Optional[List[str]] = None,
    model: Union[str, Path] = _DIR / "models" / "zh_CN-huayan-medium.onnx",
    batch_size: int = 1,
    slerp_weights: Tuple[float, ...] = (0.5,),
    length_scales: Tuple[float, ...] = (0.75, 1, 1.25),
    noise_scales: Tuple[float, ...] = (0.667,),
    noise_scale_ws: Tuple[float, ...] = (0.8,),
    max_speakers: Optional[float] = None,
    verbose: bool = False,
    auto_reduce_batch_size: bool = False,
    min_phoneme_count: Optional[int] = None,
    use_cuda: bool = False,
    **kwargs,
) -> None:
    """
    Generate synthetic speech clips, saving the clips to the specified output directory.

    Args:
        text (List[str]): The text to convert into speech. Can be either a
                          a list of strings, or a path to a file with text on each line.
        output_dir (str): The location to save the generated clips.
        max_samples (int): The maximum number of samples to generate.
        file_names (List[str]): The names to use when saving the files. Must be the same length
                                as the `text` argument, if a list.
        model (str): The path to the STT model to use for generation.
        batch_size (int): The batch size to use when generated the clips
        slerp_weights (List[float]): The weights to use when mixing speakers via SLERP.
        length_scales (List[float]): Controls the average duration/speed of the generated speech.
        noise_scales (List[float]): A parameter for overall variability of the generated speech.
        noise_scale_ws (List[float]): A parameter for the stochastic duration of words/phonemes.
        max_speakers (int): The maximum speaker number to use, if the model is multi-speaker.
        verbose (bool): Enable or disable more detailed logging messages (default: False).
        auto_reduce_batch_size (bool): Automatically and temporarily reduce the batch size
                                       if CUDA OOM errors are detected, and try to resume generation.
        min_phoneme_count (int): If set, ensure this number of phonemes is always sent to the model.
                                 Clip audio to extract original phrase.

    Returns:
        None
    """

    if max_samples is None:
        max_samples = len(text)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = f"{model}.json"
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    voice = config["espeak"]["voice"]
    sample_rate = config["audio"]["sample_rate"]
    num_speakers = config["num_speakers"]
    if max_speakers is not None:
        num_speakers = min(num_speakers, max_speakers)

    sample_idx = 0
    settings_iter = it.cycle(
        it.product(
            slerp_weights,
            length_scales,
            noise_scales,
            noise_scale_ws,
        )
    )

    speakers_iter = it.cycle(range(num_speakers))

    if isinstance(text, str) and os.path.exists(text):
        texts = it.cycle(
            [
                i.strip()
                for i in open(text, "r", encoding="utf-8").readlines()
                if len(i.strip()) > 0
            ]
        )
    elif isinstance(text, list):
        texts = it.cycle(text)
    else:
        texts = it.cycle([text])

    if file_names:
        file_names = it.cycle(file_names)
    
    voice = PiperVoice.load(model, config_path=config_path, use_cuda=use_cuda)
    
    resample_rate = 16000

    for sample_idx in trange(max_samples):
        slerp_weight, length_scale, noise_scale, noise_scale_w = next(settings_iter)
        
        from io import BytesIO
        
        wav_mem_file = BytesIO()
        wav_file = wave.open(wav_mem_file, "wb")

        voice.synthesize(
            next(texts), 
            wav_file=wav_file,
            speaker_id=next(speakers_iter),
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_scale_w
        )
        # wav_mem_file.seek(0)
        # data = wav_mem_file.read()
        # with open("./test2.wav","wb") as output:
        #     output.write(data)
        wav_mem_file.seek(0)
        audio, original_sr = librosa.load(wav_mem_file, sr=sample_rate)
        resampled_audio_data = librosa.resample(audio, orig_sr=original_sr, target_sr=resample_rate)
        
        if isinstance(file_names, it.cycle):
            wav_path = os.path.join(output_dir, next(file_names))
        else:
            wav_path = os.path.join(output_dir, f"{sample_idx}.wav")
            
        
        import soundfile as sf
        
        sf.write(str(wav_path),resampled_audio_data,16000) 

        # wav_file: wave.Wave_write = wave.open(str(wav_path), "wb")
        # with wav_file:
        #     wav_file.setframerate(resample_rate)
        #     wav_file.setsampwidth(2)
        #     wav_file.setnchannels(1)
        #     wav_file.writeframes(resampled_audio_data)

    _LOGGER.info("Done")


def remove_silence(
    x: np.ndarray,
    frame_duration: float = 0.030,
    sample_rate: int = 16000,
    min_start: int = 2000,
) -> np.ndarray:
    """Uses webrtc voice activity detection to remove silence from the clips"""
    vad = webrtcvad.Vad(0)
    if x.dtype in (np.float32, np.float64):
        x = (x * 32767).astype(np.int16)
    x_new = x[0:min_start].tolist()
    step_size = int(sample_rate * frame_duration)
    for i in range(min_start, x.shape[0] - step_size, step_size):
        vad_res = vad.is_speech(x[i : i + step_size].tobytes(), sample_rate)
        if vad_res:
            x_new.extend(x[i : i + step_size].tolist())
    return np.array(x_new).astype(np.int16)
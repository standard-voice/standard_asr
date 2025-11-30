# Copyright 2025 The Standard ASR Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal
from faster_whisper import WhisperModel  # type: ignore[stub]
from pydantic import Field

from standard_asr import BaseConfig, StandardASR


class FasterWhisperConfig(BaseConfig[Literal["faster-whisper"]]):
    """
    Configuration for the Faster-Whisper engine.

    Attributes:
        engine (Literal['faster-whisper']): Discriminator value.
        model_path (str): Local path or Hugging Face model name.
        device (str): Compute device (e.g., 'cpu', 'cuda', 'auto').
        compute_type (str): Quantization / precision hint (e.g., 'int8', 'float16').
    """

    engine: Literal["faster-whisper"] = "faster-whisper"

    model_path: str = Field(
        "base.en",
        description="Path to model or model identifier from Hugging Face Hub.",
    )
    device: str = Field(
        "auto",
        description="Device for computation (e.g., 'cpu', 'cuda', 'auto').",
    )
    compute_type: str = Field(
        "default",
        description="Quantization or precision type (e.g., 'int8', 'float16', 'default').",
    )
    download_root: str | None = Field(
        None,
        description="Directory where the models should be saved. If not set, the models are saved in the standard Hugging Face cache directory.",
    )
    language: str | None = Field(
        None,
        description="The language spoken in the audio. It should be a language code such as 'en' or 'fr'. If not set, the language will be detected in the first 30 seconds of audio.",
    )
    beam_search: int = Field(
        5,
        description="The beam size to use for beam search decoding.",
    )
    prompt: str | None = Field(
        None,
        description="A prompt to guide the transcription (optional).",
    )


class FasterWhisperASR(StandardASR):
    config: FasterWhisperConfig

    def __init__(
        self,
        model_path: str = "distil-medium.en",
        download_root: str | None = None,
        language: str | None = None,
        device: str = "auto",
        compute_type: str = "int8",
        prompt: str | None = None,
        beam_search: int = 5,
    ) -> None:
        self.config = FasterWhisperConfig(
            engine="faster-whisper",
            model_path=model_path,
            download_root=download_root,
            language=language,
            device=device,
            compute_type=compute_type,
            prompt=prompt,
            beam_search=beam_search,
        )

        self.model = WhisperModel(
            model_size_or_path=model_path,
            download_root=download_root,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio) -> str:
        segments, info = self.model.transcribe(
            audio,
            beam_size=5,
            language=self.config.language,
            condition_on_previous_text=False,
            initial_prompt=self.config.prompt,
        )
        text = [segment.text for segment in segments]

        if not text:
            return ""
        else:
            return "".join(text)

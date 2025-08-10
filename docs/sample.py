
from standard_asr import StandardASR
from faster_whisper import StandardWhisperModel

whisper_instance: StandardASR = StandardWhisperModel()

whisper_instance.transcribe(audio)
whisper_instance.transcribe_async(audio)


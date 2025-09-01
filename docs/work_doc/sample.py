from standard_asr import StandardASR
from faster_whisper import StandardWhisperModel

whisper_instance: StandardASR = StandardWhisperModel()

# Example placeholder to satisfy linters; replace with real audio bytes/array
audio = b""

whisper_instance.transcribe(audio)
whisper_instance.transcribe_async(audio)

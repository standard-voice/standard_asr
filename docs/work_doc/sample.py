from faster_whisper import StandardWhisperModel

from standard_asr import StandardASR

whisper_instance: StandardASR = StandardWhisperModel()

# Example placeholder to satisfy linters; replace with real audio bytes/array
audio = b""

whisper_instance.transcribe(audio)
whisper_instance.transcribe_async(audio, options={"language": "en-US"})
whisper_instance.transcribe_async(audio, options=whisper_instance.options(language="en-US"))

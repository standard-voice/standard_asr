# Standard ASR

> ⚠️⚠️⚠️ Standard ASR is still work in progress!! Breaking changes may be introduced at any moment!!
> 
> For production use, please wait until `v1.0.0` release, where we will be stabilizing the APIs and enforce migration policy when breaking changes do happen. We strictly follow semantic versioning.
>
> Please test out standard library and give us feedback or your opinion. Let's shape the future of ASR library together!

![standard_asr_concept](docs/assets/concept.jpg)


## Introduction
Standard ASR (Automatic Speech Recognition) is a protocol that attempts to standardize the way to interact with different ASR models. 

Think of this as the USB-C for speech recognition libraries. We help standardize how the users of ASR libraries interact with ASR libraries, so application developers can use one code to interact with different ASR packages and models.

ASR integration code should be written once and only once. Application developers should not be writing new code when new ASR packages got released. One code should work with any ASR models, because they all do one thing: transcribe audio into text.

That's what we tries to do: the usb protocol for ASR libraries.



---



- Pydantic v2 to model ASR's settings
- Fully async support
- pytest
- use logging


Core mission
- provide standardized way to interact with different ASR models
- AI native: AI knows (or can easily know) how to use `standard_asr`


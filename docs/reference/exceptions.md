# standard_asr.contract.exceptions

The domain exceptions the library raises. All inherit from `StandardASRError`,
so `except StandardASRError` catches every Standard ASR domain error while
letting other exceptions propagate. Plain caller misuse — for example, passing
two mutually exclusive arguments — raises the built-in `ValueError` or
`TypeError` instead. Domain-typed misuse (`InvalidSessionUseError`,
`InvalidProviderParamError`) stays inside the hierarchy.

::: standard_asr.contract.exceptions

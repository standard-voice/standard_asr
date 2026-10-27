# SPDX-FileCopyrightText: The Standard ASR Authors
# SPDX-License-Identifier: Apache-2.0

"""The endpoint check of ``standard-asr doctor --network``.

huggingface_hub is replaced by fake modules in ``sys.modules``, and its
client by fakes of both families it can return: an httpx ``Client`` (1.x)
and a requests ``Session`` (0.x). Most failures are real standard-library
exceptions, chained the way the clients chain them, so the classification
reads the cause chains it reads in production; the others are stand-ins
named after the libraries' own exceptions. No test sends a request.
"""

from __future__ import annotations

import socket
import ssl
import sys
import threading
import types
from typing import Any

import pytest

from standard_asr.toolchain import doctor

ENDPOINT = "https://huggingface.co"

#: Text that stands for an error's message: no result may contain it.
MESSAGE = "MESSAGE-TEXT"

#: A password, or the part of one, that a client's error message can quote.
SECRET = "SENTINEL"


class _Response:
    def __init__(self, status_code: object) -> None:
        self.status_code = status_code
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _HttpxClient:
    """Stands for ``httpx.Client``: the check knows the family by class name and module."""

    def __init__(self, answer: object) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []
        self.response: _Response | None = None

    def head(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        if isinstance(self.answer, BaseException):
            raise self.answer
        self.response = _Response(self.answer)
        return self.response


_HttpxClient.__name__ = _HttpxClient.__qualname__ = "Client"
_HttpxClient.__module__ = "httpx._client"


class _RequestsSession(_HttpxClient):
    """Stands for ``requests.Session``."""


_RequestsSession.__name__ = _RequestsSession.__qualname__ = "Session"
_RequestsSession.__module__ = "requests.sessions"


class _AppClient(_HttpxClient):
    """An application's own subclass, set with ``set_client_factory()``."""


_AppClient.__module__ = "myapp.http"


#: Stands for ``httpx.InvalidURL``, whose message quotes the part of a URL it rejected.
_InvalidURL = type("InvalidURL", (Exception,), {"__module__": "httpx"})


class _SecretStatus:
    """A status code that cannot be read, with an error message that quotes a secret."""

    def __int__(self) -> int:
        raise ValueError(f"cannot read {SECRET}")


def _install_hub(
    monkeypatch: pytest.MonkeyPatch, session: object, *, endpoint: str | None = ENDPOINT
) -> None:
    hub = types.ModuleType("huggingface_hub")
    hub.__version__ = "1.19.0"  # type: ignore[attr-defined]

    def get_session() -> object:
        if isinstance(session, BaseException):
            raise session
        return session

    utils = types.ModuleType("huggingface_hub.utils")
    utils.get_session = get_session  # type: ignore[attr-defined]
    constants = types.ModuleType("huggingface_hub.constants")
    if endpoint is not None:
        constants.ENDPOINT = endpoint  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)


@pytest.fixture(autouse=True)
def _no_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    # The check reports the proxies of the process; every test starts with none.
    monkeypatch.setattr(doctor.urllib.request, "getproxies", dict)


def _wrapped(cause: BaseException, name: str = "ConnectError") -> BaseException:
    """Return ``cause`` wrapped in an exception class called ``name``, chained as a client does."""
    wrapper = type(name, (Exception,), {})
    try:
        try:
            raise cause
        except BaseException as inner:
            raise wrapper(str(cause)) from inner
    except BaseException as error:
        return error


def test_an_http_answer_is_reachable_and_one_request_follows_no_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _HttpxClient(302)
    _install_hub(monkeypatch, client)
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == "reachable"
    assert check.status_code == 302
    assert check.is_problem is False
    assert check.client == "huggingface_hub 1.19.0 (httpx)"
    assert check.endpoint == ENDPOINT
    assert client.calls == [{"url": ENDPOINT, "timeout": 1, "follow_redirects": False}]
    assert client.response is not None and client.response.closed


def test_a_requests_session_gets_the_requests_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _RequestsSession(200)
    _install_hub(monkeypatch, session)
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == "reachable"
    assert check.client == "huggingface_hub 1.19.0 (requests)"
    assert session.calls == [{"url": ENDPOINT, "timeout": 1, "allow_redirects": False}]


def test_an_application_subclass_is_driven_as_its_family(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_hub(monkeypatch, _AppClient(200))
    assert doctor.check_endpoint(timeout=1).client == "huggingface_hub 1.19.0 (httpx)"


@pytest.mark.parametrize("status", [401, 404, 429])
def test_an_answer_that_does_not_show_a_blocked_path_is_reachable(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    # A mirror can answer 401 or 404 to a request for its root without a token.
    # A 429 limits this client's rate, and Hugging Face limits page requests
    # apart from downloads, so it does not show that a download fails.
    _install_hub(monkeypatch, _HttpxClient(status))
    check = doctor.check_endpoint(timeout=1)
    assert (check.outcome, check.status_code, check.is_problem) == ("reachable", status, False)


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (403, "forbidden"),
        (407, "proxy_refused"),
        (451, "forbidden"),
        (500, "server_error"),
        (503, "server_error"),
    ],
)
def test_a_refusing_or_failing_answer_is_a_problem(
    monkeypatch: pytest.MonkeyPatch, status: int, outcome: str
) -> None:
    _install_hub(monkeypatch, _HttpxClient(status))
    check = doctor.check_endpoint(timeout=1)
    assert (check.outcome, check.status_code, check.is_problem) == (outcome, status, True)
    assert check.remedy


@pytest.mark.parametrize(("status", "mentions_token"), [(403, True), (451, False)])
def test_only_the_403_remedy_suggests_a_download_with_the_token(
    monkeypatch: pytest.MonkeyPatch, status: int, mentions_token: bool
) -> None:
    # A mirror can refuse a request without a token with 403, never with 451.
    _install_hub(monkeypatch, _HttpxClient(status))
    remedy = doctor.check_endpoint(timeout=1).remedy
    assert remedy is not None
    assert ("token" in remedy) is mentions_token


@pytest.mark.parametrize(
    ("error", "outcome", "remedy"),
    [
        (
            type("OfflineModeIsEnabled", (ConnectionError,), {})(MESSAGE),
            "offline",
            "HF_HUB_OFFLINE",
        ),
        (
            _wrapped(ssl.SSLCertVerificationError(1, MESSAGE)),
            "certificate_not_verified",
            "SSL_CERT_FILE",
        ),
        (_wrapped(socket.gaierror(8, MESSAGE)), "name_not_resolved", "host name"),
        (_wrapped(ConnectionRefusedError(61, MESSAGE)), "connection_refused", "port"),
        (_wrapped(TimeoutError(MESSAGE), "ReadTimeout"), "timed_out", "HTTPS_PROXY"),
        (type("ConnectTimeout", (Exception,), {})(MESSAGE), "timed_out", "HTTPS_PROXY"),
        (_wrapped(OSError(MESSAGE), "ProxyError"), "proxy_refused", "NO_PROXY"),
        (RuntimeError(MESSAGE), "failed", "get_session().head(constants.ENDPOINT)"),
    ],
)
def test_a_failed_request_is_classified_from_its_causes(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, outcome: str, remedy: str
) -> None:
    _install_hub(monkeypatch, _HttpxClient(error))
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == outcome
    assert check.is_problem
    assert check.remedy is not None and remedy in check.remedy
    # The result names the error by its type, never by its message.
    assert MESSAGE not in check.detail
    if outcome != "offline":
        assert type(error).__qualname__ in check.detail


@pytest.mark.parametrize(
    ("code", "message", "reason"),
    [
        (
            19,
            "self-signed certificate in certificate chain",
            "self-signed certificate in certificate chain",
        ),
        # The ``ssl`` module adds the host the client connected to, which a
        # misread URL can take from a password: the report leaves it out.
        (
            64,
            f"IP address mismatch, certificate is not valid for '{SECRET}'.",
            "IP address mismatch",
        ),
        (62, f"Hostname mismatch, certificate is not valid for '{SECRET}'.", "Hostname mismatch"),
    ],
)
def test_a_certificate_failure_gives_openssls_description(
    monkeypatch: pytest.MonkeyPatch, code: int, message: str, reason: str
) -> None:
    cause = ssl.SSLCertVerificationError(1, MESSAGE)
    cause.verify_code = code
    cause.verify_message = message
    _install_hub(monkeypatch, _HttpxClient(_wrapped(cause)))
    check = doctor.check_endpoint(timeout=1)
    assert check.detail.startswith(f"The TLS certificate did not verify: {reason} (")
    assert check.detail.endswith("ConnectError, caused by ssl.SSLCertVerificationError).")
    assert SECRET not in check.detail
    assert MESSAGE not in check.detail


def test_the_certificate_remedy_names_the_variable_of_the_client_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _wrapped(ssl.SSLCertVerificationError(1, "certificate verify failed"), "SSLError")
    _install_hub(monkeypatch, _RequestsSession(error))
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == "certificate_not_verified"
    assert check.remedy is not None and "REQUESTS_CA_BUNDLE" in check.remedy


def test_the_check_does_not_apply_without_huggingface_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules makes the import fail as if the package were absent.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == "not_applicable"
    assert check.is_problem is False
    assert check.endpoint is None


@pytest.mark.parametrize(
    "failure",
    [ImportError("broken install"), ModuleNotFoundError("No module named 'tqdm'", name="tqdm")],
)
def test_a_broken_installation_is_not_an_absence(
    monkeypatch: pytest.MonkeyPatch, failure: ImportError
) -> None:
    real_import = doctor.importlib.import_module

    def import_module(name: str, package: str | None = None) -> types.ModuleType:
        if name == "huggingface_hub":
            raise failure
        return real_import(name, package)

    monkeypatch.setattr(doctor.importlib, "import_module", import_module)
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == "could_not_run"
    assert "failed to import" in check.detail
    assert str(failure) not in check.detail
    # The command names the interpreter that runs the check.
    assert check.remedy is not None
    assert check.remedy.startswith(f'Run {sys.executable} -c "import huggingface_hub"')


def test_a_client_that_cannot_be_created_or_driven_could_not_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_hub(monkeypatch, FileNotFoundError(2, "No such file or directory"))
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == "could_not_run"
    assert check.remedy is not None and "SSL_CERT_FILE" in check.remedy

    _install_hub(monkeypatch, object())
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == "could_not_run"
    assert "httpx or a requests client" in check.detail

    _install_hub(monkeypatch, _HttpxClient(200), endpoint=None)
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == "could_not_run"
    assert "cannot be used" in check.detail

    # A submodule that fails to import, as when ``httpx`` is missing, is reported,
    # and huggingface_hub's lazy loader never gets the chance to print it.
    _install_hub(monkeypatch, _HttpxClient(200))
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", None)
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == "could_not_run"
    assert "cannot be used" in check.detail


def test_an_unexpected_answer_fails_instead_of_waiting_for_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _HttpxClient("not a status")
    _install_hub(monkeypatch, client)
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == "failed"
    assert "The check failed" in check.detail
    assert client.response is not None and client.response.closed


def test_the_deadline_covers_a_request_that_never_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    # A stalled name lookup blocks before the request's own timeout starts;
    # the call returns at the deadline and leaves the thread behind.
    release = threading.Event()

    class Stalled(_HttpxClient):
        def head(self, url: str, **kwargs: Any) -> _Response:
            release.wait(5)
            return _Response(200)

    Stalled.__name__ = "Client"
    _install_hub(monkeypatch, Stalled(200))
    try:
        check = doctor.check_endpoint(timeout=0.05)
    finally:
        release.set()
    assert check.outcome == "timed_out"
    assert "within 0.05 seconds" in check.detail
    # The client is created before the wait starts, so the report names it.
    assert check.client == "huggingface_hub 1.19.0 (httpx)"


def test_credentials_never_reach_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``httpx`` accepts a space in a password, so the fields are masked by their
    # structure, not by a pattern that stops at whitespace.
    monkeypatch.setattr(
        doctor.urllib.request,
        "getproxies",
        lambda: {
            "https": "http://user:sec ret@proxy:8080",
            "no": "localhost,http://alice:LEFT,RIGHT@m.example,127.0.0.1",
        },
    )
    # The error's message quotes the URL; the result names the error by its
    # type and never quotes the message.
    error = _wrapped(
        ConnectionRefusedError(61, "refused " + "x" * 360 + " http://user:sec-ret@mirror/api")
    )
    _install_hub(monkeypatch, _HttpxClient(error), endpoint="http://user:sec ret@mirror")
    check = doctor.check_endpoint(timeout=1)
    rendered = doctor.format_report(doctor.DoctorReport(python_version="3.13", endpoint=check))
    assert "sec" not in rendered
    assert check.endpoint == "http://***@mirror"
    # A URL pasted into the no-proxy list is masked whole, though its
    # password holds the comma that separates the list's hosts.
    assert check.proxies == {"https": "http://***@proxy:8080", "no": "***@m.example,127.0.0.1"}
    assert "proxies: https=http://***@proxy:8080" in rendered
    assert "no proxy for: ***@m.example,127.0.0.1" in rendered
    for piece in ("alice", "LEFT", "RIGHT"):
        assert piece not in rendered
        assert piece not in repr(check)


def test_the_command_quotes_an_interpreter_path_with_a_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.sys, "executable", "/Applications/My Python/bin/python3")
    _install_hub(monkeypatch, _HttpxClient(RuntimeError(MESSAGE)))
    check = doctor.check_endpoint(timeout=1)
    assert check.remedy is not None
    assert check.remedy.startswith('Run "/Applications/My Python/bin/python3" -c "from ')


@pytest.mark.parametrize("where", ["import", "settings", "client", "request", "answer"])
def test_no_error_message_reaches_the_result(monkeypatch: pytest.MonkeyPatch, where: str) -> None:
    # A password with a raw "/" makes ``httpx`` reject the URL with "Invalid port:
    # 'SENTINEL'": part of the password, with no URL around it for a
    # redaction rule to find. So the result names every error by its type.
    rejected = _InvalidURL(f"Invalid port: '{SECRET}'")
    real_import = doctor.importlib.import_module

    def import_module(name: str, package: str | None = None) -> types.ModuleType:
        if (where, name) == ("import", "huggingface_hub"):
            raise ImportError(f"broken by {SECRET}")
        if (where, name) == ("settings", "huggingface_hub.utils"):
            raise RuntimeError(f"broken by {SECRET}")
        return real_import(name, package)

    clients = {"client": rejected, "request": _HttpxClient(rejected)}
    _install_hub(monkeypatch, clients.get(where, _HttpxClient(_SecretStatus())))
    monkeypatch.setattr(doctor.importlib, "import_module", import_module)
    check = doctor.check_endpoint(timeout=1)
    rendered = doctor.format_report(doctor.DoctorReport(python_version="3.13", endpoint=check))
    assert check.is_problem
    assert SECRET not in repr(check)
    assert SECRET not in rendered
    named = {
        "import": "(ImportError).",
        "settings": "(RuntimeError).",
        "client": "(httpx.InvalidURL).",
        "request": "(httpx.InvalidURL).",
        "answer": "(ValueError).",
    }
    assert check.detail.endswith(named[where])


@pytest.mark.parametrize(
    ("proxy", "endpoint", "ambiguous"),
    [
        # A raw "/", "?", or "#" in a password ends the host part early.
        ("http://alice:SEN/TINEL@proxy:8080", ENDPOINT, ("https",)),
        ("http://alice:p?w@proxy:8080", ENDPOINT, ("https",)),
        # A value that does not start with a scheme is read from its start,
        # even with a "://" in its password.
        ("alice:p#w@proxy:8080", ENDPOINT, ("https",)),
        ("alice:SE://CRET@proxy:8080", ENDPOINT, ("https",)),
        # A raw "@" in a password, or a "/" after the last "@", reads as meant.
        ("http://alice:p@w@proxy:8080", ENDPOINT, ()),
        ("http://alice:pw@proxy:8080/", ENDPOINT, ()),
        ("http://proxy:8080", "https://alice:a/b@mirror", ("endpoint",)),
        # An "@" in a path looks the same, which is why the note says "if".
        ("http://proxy:8080", "https://mirror/models@v2", ("endpoint",)),
    ],
)
def test_a_url_a_client_can_read_differently_from_the_report_is_named(
    monkeypatch: pytest.MonkeyPatch, proxy: str, endpoint: str, ambiguous: tuple[str, ...]
) -> None:
    # The hosts in NO_PROXY are not a URL, so they are never named.
    proxies = {"https": proxy, "no": "a/b@c"}
    monkeypatch.setattr(doctor.urllib.request, "getproxies", lambda: proxies)
    _install_hub(monkeypatch, _HttpxClient(200), endpoint=endpoint)
    check = doctor.check_endpoint(timeout=1)
    assert check.ambiguous_urls == ambiguous
    rendered = doctor.format_report(doctor.DoctorReport(python_version="3.13", endpoint=check))
    notes = [line.strip() for line in rendered.splitlines() if "before its last '@'" in line]
    names = [
        "The endpoint URL" if name == "endpoint" else f"The {name} proxy URL" for name in ambiguous
    ]
    assert [note.split(" has ", 1)[0] for note in notes] == [f"[INFO] {name}" for name in names]


def test_the_verdict_and_the_report_keep_the_two_checks_apart() -> None:
    report = doctor.DoctorReport(python_version="3.13")
    assert report.is_clean and report.dependencies_clean
    rendered = doctor.format_report(report)
    assert "No dependency conflicts detected." in rendered
    assert "Network: not checked." in rendered

    failed = doctor.EndpointCheck(
        "timed_out", "http://mirror", "huggingface_hub 1.19.0 (httpx)", None, "slow", "fix it", {}
    )
    report = doctor.DoctorReport(python_version="3.13", endpoint=failed)
    # A network problem does not make the dependency analysis look incomplete.
    assert report.dependencies_clean and not report.is_clean
    rendered = doctor.format_report(report)
    assert "No dependency conflicts detected." in rendered
    assert "[FAIL] slow" in rendered and "fix: fix it" in rendered
    assert "proxies: none" in rendered
    assert "plain HTTP" in rendered
    assert doctor.NOT_CHECKED in rendered

    absent = doctor.EndpointCheck("not_applicable", None, None, None, "not installed", None, {})
    report = doctor.DoctorReport(python_version="3.13", endpoint=absent)
    assert report.is_clean
    assert "[INFO] not installed" in doctor.format_report(report)

    not_run = doctor.EndpointCheck("could_not_run", None, None, None, "broken", None, {})
    rendered = doctor.format_report(doctor.DoctorReport(python_version="3.13", endpoint=not_run))
    assert "[FAIL] broken" in rendered
    assert doctor.NOT_CHECKED not in rendered

    fine = doctor.EndpointCheck("reachable", ENDPOINT, "x", 200, "ok", None, {})
    rendered = doctor.format_report(doctor.DoctorReport(python_version="3.13", endpoint=fine))
    assert "[OK] ok" in rendered
    assert "plain HTTP" not in rendered


def test_diagnose_runs_the_check_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []
    fine = doctor.EndpointCheck("reachable", ENDPOINT, "x", 200, "ok", None, {})

    def check_endpoint() -> doctor.EndpointCheck:
        calls.append(None)
        return fine

    def no_plugins(*, group: str) -> list[object]:
        return []

    monkeypatch.setattr(doctor, "check_endpoint", check_endpoint)
    monkeypatch.setattr(doctor, "entry_points", no_plugins)
    assert doctor.diagnose().endpoint is None
    assert calls == []
    assert doctor.diagnose(network=True).endpoint is fine
    assert calls == [None]

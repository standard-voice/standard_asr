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


@pytest.mark.parametrize(
    ("status", "outcome"), [(407, "proxy_refused"), (500, "server_error"), (503, "server_error")]
)
def test_a_407_or_a_5xx_answer_is_a_problem(
    monkeypatch: pytest.MonkeyPatch, status: int, outcome: str
) -> None:
    _install_hub(monkeypatch, _HttpxClient(status))
    check = doctor.check_endpoint(timeout=1)
    assert (check.outcome, check.status_code, check.is_problem) == (outcome, status, True)
    assert check.remedy


@pytest.mark.parametrize(
    ("error", "outcome", "remedy"),
    [
        (
            type("OfflineModeIsEnabled", (ConnectionError,), {})("offline"),
            "offline",
            "HF_HUB_OFFLINE",
        ),
        (
            _wrapped(ssl.SSLCertVerificationError(1, "certificate verify failed")),
            "certificate_not_verified",
            "SSL_CERT_FILE",
        ),
        (_wrapped(socket.gaierror(8, "nodename nor servname")), "name_not_resolved", "host name"),
        (_wrapped(ConnectionRefusedError(61, "refused")), "connection_refused", "port"),
        (_wrapped(TimeoutError("timed out"), "ReadTimeout"), "timed_out", "HTTPS_PROXY"),
        (type("ConnectTimeout", (Exception,), {})("slow"), "timed_out", "HTTPS_PROXY"),
        (
            _wrapped(OSError("Tunnel connection failed: 403"), "ProxyError"),
            "proxy_refused",
            "NO_PROXY",
        ),
        (RuntimeError("something else"), "failed", None),
    ],
)
def test_a_failed_request_is_classified_from_its_causes(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, outcome: str, remedy: str | None
) -> None:
    _install_hub(monkeypatch, _HttpxClient(error))
    check = doctor.check_endpoint(timeout=1)
    assert check.outcome == outcome
    assert check.is_problem
    if remedy is None:
        assert check.remedy is None
        assert "something else" in check.detail
    else:
        assert check.remedy is not None and remedy in check.remedy


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
        lambda: {"https": "http://user:sec ret@proxy:8080", "no": "localhost"},
    )
    # A long message that quotes the URL: the summary removes the credentials
    # before it cuts the message at its length bound.
    error = _wrapped(
        ConnectionRefusedError(61, "refused " + "x" * 360 + " http://user:sec-ret@mirror/api")
    )
    _install_hub(monkeypatch, _HttpxClient(error), endpoint="http://user:sec ret@mirror")
    check = doctor.check_endpoint(timeout=1)
    rendered = doctor.format_report(doctor.DoctorReport(python_version="3.13", endpoint=check))
    assert "sec" not in rendered
    assert check.endpoint == "http://***@mirror"
    assert check.proxies == {"https": "http://***@proxy:8080", "no": "localhost"}
    assert "proxies: https=http://***@proxy:8080" in rendered
    assert "no proxy for: localhost" in rendered


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

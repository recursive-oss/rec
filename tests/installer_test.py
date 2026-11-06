#!/usr/bin/env python3

import functools
import hashlib
import http.server
import os
import subprocess
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "install.sh"
ARTIFACT = "rec-darwin-arm64"


def fake_rec(version):
    return (
        "#!/bin/sh\n"
        'case "$1" in\n'
        f'  --version) echo "rec {version}";;\n'
        '  doctor) echo doctor >> "$HOME/doctor.log"; [ -z "${TEST_BAD_DEVICE:-}" ];;\n'
        '  status) [ -f "$HOME/running" ];;\n'
        f'  stop) echo "rec {version}" >> "$HOME/stop.log"; rm "$HOME/running";;\n'
        '  serve) echo "$*" >> "$HOME/serve.log";;\n'
        "  *) exit 2;;\n"
        "esac\n"
    )


def publish(directory, version, reported=None, bad_hash=False):
    directory.mkdir(parents=True, exist_ok=True)
    body = fake_rec(reported or version).encode()
    (directory / ARTIFACT).write_bytes(body)
    digest = "0" * 64 if bad_hash else hashlib.sha256(body).hexdigest()
    (directory / (ARTIFACT + ".meta")).write_text(f"{digest} {len(body)}\n")


FIXTURES = {
    "uname": '#!/bin/sh\ncase "$1" in -s) echo "${TEST_OS:-Darwin}";; -m) echo "${TEST_ARCH:-arm64}";; esac\n',
    "curl": "#!/bin/sh\nexit 22\n",
}


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    with tempfile.TemporaryDirectory(prefix="rec-install-", dir="/tmp") as td:
        root = Path(td)
        releases = root / "releases"
        publish(releases / "download/v0.3.0", "0.3.0")
        publish(releases / "latest/download", "0.3.0")
        publish(releases / "download/v0.2.0", "0.2.0")
        publish(releases / "download/v0.4.0", "0.4.0", reported="0.5.0")
        publish(releases / "download/v0.6.0", "0.6.0", bad_hash=True)
        bins = root / "bin"
        offline = root / "offline"
        for directory, names in [(bins, ["uname"]), (offline, ["uname", "curl"])]:
            directory.mkdir()
            for name in names:
                (directory / name).write_text(FIXTURES[name])
                (directory / name).chmod(0o755)
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(Quiet, directory=str(releases))
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}"
        counter = 0

        def new_home(extra=None):
            nonlocal counter
            counter += 1
            home = root / f"home {counter}"
            home.mkdir()
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("REC_", "XDG_", "ZDOTDIR", "TEST_"))
            }
            env.update(
                HOME=str(home),
                SHELL="/bin/bash",
                PATH=f"{bins}:/usr/bin:/bin:/usr/sbin:/sbin",
                REC_RELEASES_URL=url,
                REC_INSTALL_ALLOW_HTTP="1",
            )
            env.update(extra or {})
            return home, env

        def install(env, *args, ok=True):
            r = subprocess.run(
                ["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True
            )
            assert (r.returncode == 0) == ok, r.stdout + r.stderr
            assert not (Path(env["HOME"]) / ".local/share/rec/recursive_rec.lock").exists()
            return r

        def installed(home):
            link = home / ".local/bin/rec"
            return link.resolve() if link.is_symlink() else None

        home, env = new_home()
        install(env)
        first = installed(home)
        assert first == (home / ".local/share/rec/releases/0.3.0/rec").resolve(), first
        assert (home / "doctor.log").read_text() == "doctor\n"
        assert not (home / "serve.log").exists(), "installer started the model by default"
        install(env)
        assert installed(home) == first
        assert (home / ".bashrc").read_text().count("# Recursive CLI PATH") == 1

        install(dict(env, REC_VERSION="v0.2.0"))
        assert installed(home).parent.name == "0.2.0"
        for extra, reason in [
            ({"REC_VERSION": "0.4.0"}, "expected 0.4.0"),
            ({"REC_VERSION": "0.6.0"}, "checksum"),
            ({"REC_VERSION": "9.9.9"}, "download failed"),
            ({"REC_INSTALL_ALLOW_HTTP": ""}, "REC_INSTALL_ALLOW_HTTP"),
            ({"REC_RELEASES_URL": "http://example.com"}, "HTTPS"),
            ({"TEST_BAD_DEVICE": "1"}, "requirements"),
            ({"REC_VERSION": "../x"}, "invalid REC_VERSION"),
        ]:
            r = install(dict(env, **extra), ok=False)
            assert reason in r.stderr, (extra, r.stderr)
            assert installed(home).parent.name == "0.2.0"

        (home / "running").touch()
        r = install(env)
        assert "Stopped the running model" in r.stdout
        assert (home / "stop.log").read_text() == "rec 0.2.0\n"
        assert not (home / "running").exists()
        assert installed(home).parent.name == "0.3.0"

        home, env = new_home()
        install(env, "--serve")
        assert (home / "serve.log").read_text() == "serve\n"

        for os_name, arch in [("Linux", "aarch64"), ("Darwin", "x86_64")]:
            home, env = new_home({"TEST_OS": os_name, "TEST_ARCH": arch})
            assert "Apple Silicon" in install(env, ok=False).stderr
            assert installed(home) is None

        home, env = new_home()
        del env["REC_RELEASES_URL"]
        r = install(dict(env, PATH=f"{offline}:/usr/bin:/bin:/usr/sbin:/sbin"), ok=False)
        public = "https://github.com/recursive-oss/rec/releases/latest/download/rec-darwin-arm64"
        assert f"download failed: {public}" in r.stderr, r.stderr

        home, env = new_home()
        (home / ".local/bin").mkdir(parents=True)
        (home / ".local/bin/rec").write_text("unrelated program\n")
        install(env, ok=False)
        assert (home / ".local/bin/rec").read_text() == "unrelated program\n"

        other = root / "other program"
        other.mkdir()
        (other / "rec").write_text('#!/bin/sh\necho "$*" >> "$HOME/other.log"\n')
        (other / "rec").chmod(0o755)
        notice = "Other rec commands"
        listed = f"  {other}/rec\n"

        # A conflict ahead of the install warns on the first install and on later updates.
        home, env = new_home()
        env["PATH"] = f"{other}:{env['PATH']}"
        for _ in range(2):
            r = install(env)
            assert notice in r.stderr and listed in r.stderr, r.stderr
            assert installed(home).parent.name == "0.3.0"
        install(env, "--serve")
        assert (home / "serve.log").read_text() == "serve\n"
        assert not (home / "other.log").exists(), "installer ran the other program"

        # A conflict behind the install is still listed, once per path, without the install itself.
        home, env = new_home()
        own = home / ".local/bin"
        env["PATH"] = f"{own}/:{other}:{other}:{env['PATH']}"
        r = install(env)
        assert r.stderr.count(listed) == 1, r.stderr
        assert f"  {own}/" not in r.stderr, r.stderr

        # The install alone is not a conflict, however PATH spells its directory.
        home, env = new_home()
        assert notice not in install(env).stderr
        own = home / ".local/bin"
        for entry in [f"{own}", f"{own}/", f"{own}/."]:
            r = install(dict(env, PATH=f"{entry}:{env['PATH']}"))
            assert notice not in r.stderr, (entry, r.stderr)
        server.shutdown()
        server.server_close()
        print(
            "Installer: public default URL, pins, idempotence, checksum/version/device rejection, running-server stop, --serve, platform checks, collision preservation and other rec warnings passed"
        )


if __name__ == "__main__":
    main()

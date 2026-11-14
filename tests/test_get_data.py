"""데이터 내려받기: 네트워크 없이 가짜 목록·다운로드로 이어 받기와 해시 검증만 본다."""

import hashlib

import pytest

from ampcap import get_data as D


def test_egdb_resumes_and_checks_md5(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "EGDB_N", 3)
    out = tmp_path / "EGDB" / "audio_DI"
    out.mkdir(parents=True)
    (out / "1.wav").write_bytes(b"old")
    calls = []

    def download(fid, path):
        calls.append(fid)
        if fid == "cut":
            path.write_bytes(b"half")  # 끊긴 다운로드
            return False
        path.write_bytes(b"wav")
        return True

    ok = hashlib.md5(b"wav").hexdigest()
    files = [("1.wav", "a", ok), ("2.wav", "b", ok), ("3.wav", "cut", ok), ("4.wav", "bad", "0" * 32)]
    assert D.get_egdb(tmp_path, files=files, download=download) == 2
    assert calls == ["b", "cut", "bad"]               # 있는 파일은 건너뜀
    assert (out / "1.wav").read_bytes() == b"old"
    assert not (out / "3.wav").exists() and not (out / "4.wav").exists()  # 끊김·해시 불일치는 남기지 않음
    assert not list(out.glob("*.part"))

    assert D.get_egdb(tmp_path, files=[("3.wav", "c", ok)], download=download) == 3


def test_egdb_manifest_is_complete():
    rows = D.egdb_manifest()
    assert len(rows) == D.EGDB_N == len({n for n, _, _ in rows})
    assert all(n.endswith(".wav") and len(m) == 32 for n, _, m in rows)


def test_nam_input_rejects_wrong_file(tmp_path):
    def download(fid, path):
        path.write_bytes(b"not the nam input")
        return True

    with pytest.raises(RuntimeError, match="input.wav"):
        D.get_nam_input(tmp_path, download=download)
    assert not (tmp_path / "nam" / "v3_0_0.wav").exists()

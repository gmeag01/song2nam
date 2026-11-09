"""데이터 내려받기: 네트워크 없이 가짜 목록·다운로드로 이어 받기와 해시 검증만 본다."""

import pytest

from ampcap import get_data as D


def test_egdb_resumes_and_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "EGDB_N", 3)
    out = tmp_path / "EGDB" / "audio_DI"
    out.mkdir(parents=True)
    (out / "1.wav").write_bytes(b"old")
    calls = []

    def download(fid, path):
        calls.append(fid)
        if fid == "bad":
            path.write_bytes(b"half")  # 끊긴 다운로드
            return False
        path.write_bytes(b"wav")
        return True

    lister = lambda _: [("1.wav", "a"), ("2.wav", "b"), ("3.wav", "bad"), ("README.txt", "r")]
    assert D.get_egdb(tmp_path, lister=lister, download=download) == 2
    assert calls == ["b", "bad"]                      # 있는 파일·wav 아닌 파일은 건너뜀
    assert (out / "1.wav").read_bytes() == b"old"
    assert not (out / "3.wav").exists() and not list(out.glob("*.part"))

    lister2 = lambda _: [("3.wav", "c")]
    assert D.get_egdb(tmp_path, lister=lister2, download=download) == 3


def test_nam_input_rejects_wrong_file(tmp_path):
    def download(fid, path):
        path.write_bytes(b"not the nam input")
        return True

    with pytest.raises(RuntimeError, match="input.wav"):
        D.get_nam_input(tmp_path, download=download)
    assert not (tmp_path / "nam" / "v3_0_0.wav").exists()

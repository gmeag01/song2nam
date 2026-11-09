"""데이터 내려받기: EGDB audio_DI, NAM 표준 입력 v3_0_0.wav (설계 문서 4.6 배포).

    python -m ampcap.get_data [--data data]

두 파일 모두 재배포하지 않고 공식 Google Drive에서 직접 받는다. 이미 받은 파일은 건너뛴다.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

# EGDB 공식 폴더(데모 페이지 링크, 1h9DrB4dk4QstgjNaHh7lL7IMeKdYw82_) 안의 audio_DI
EGDB_DI_FOLDER = "1AfpAaWnBvrTLHxe2DafgQ4mz4GGBjwAa"
EGDB_N = 240
EGDB_PAGE = "https://ss12f32v.github.io/Guitar-Transcription/"
# NAM 문서의 표준 입력 input.wav (v3.0.0)
NAM_INPUT_ID = "1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G"
NAM_INPUT_MD5 = "36cd1af62985c2fac3e654333e36431e"  # nam.train.core._detect_input_version의 v3.0.0 해시
NAM_DOCS = "https://neural-amp-modeler.readthedocs.io/en/latest/tutorials/colab.html"


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _gdown_file(file_id: str, out: Path) -> bool:
    import gdown

    return gdown.download(id=file_id, output=str(out), quiet=True) is not None


def get_nam_input(data_dir="data", download=_gdown_file) -> Path:
    out = Path(data_dir) / "nam" / "v3_0_0.wav"
    if out.exists() and md5(out) == NAM_INPUT_MD5:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    if not download(NAM_INPUT_ID, out) or md5(out) != NAM_INPUT_MD5:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"NAM 표준 입력을 받지 못했거나 파일이 다릅니다. {NAM_DOCS} 의 input.wav를 받아 {out}에 두세요.")
    return out


def list_drive_folder(folder_id: str) -> list[tuple[str, str]]:
    """(이름, 파일 ID) 목록. Colab에서는 Drive API로 전부, 그 밖에서는 gdown 공개 목록(최대 50개)."""
    try:
        from google.colab import auth
        from googleapiclient.discovery import build
    except ImportError:
        import gdown

        fs = gdown.download_folder(id=folder_id, skip_download=True, quiet=True, remaining_ok=True)
        return [(Path(f.path).name, f.id) for f in fs or []]
    auth.authenticate_user()
    svc = build("drive", "v3", cache_discovery=False)
    files, token = [], None
    while True:
        r = svc.files().list(q=f"'{folder_id}' in parents and trashed=false", pageSize=1000, pageToken=token,
                             fields="nextPageToken, files(id, name)", supportsAllDrives=True,
                             includeItemsFromAllDrives=True).execute()
        files += [(f["name"], f["id"]) for f in r["files"]]
        token = r.get("nextPageToken")
        if not token:
            return files


def get_egdb(data_dir="data", lister=list_drive_folder, download=_gdown_file) -> int:
    """data/EGDB/audio_DI/*.wav를 채우고 받은 파일 수를 돌려준다. 중간에 끊겨도 다시 돌리면 이어 받는다."""
    out = Path(data_dir) / "EGDB" / "audio_DI"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((n, i) for n, i in lister(EGDB_DI_FOLDER) if n.lower().endswith(".wav"))
    failed = []
    for name, fid in files:
        p = out / name
        if p.exists() and p.stat().st_size > 0:
            continue
        part = p.with_name(p.name + ".part")  # 끊긴 파일이 완성본으로 남지 않게
        if download(fid, part):
            part.replace(p)
        else:
            part.unlink(missing_ok=True)
            failed.append(name)
    n = len(list(out.glob("*.wav")))
    if failed:
        print(f"EGDB {len(failed)}개 실패({', '.join(failed[:5])}...). 잠시 뒤 다시 실행하면 이어 받습니다.")
    if n < EGDB_N:
        print(f"EGDB DI {n}/{EGDB_N}개. 나머지는 {EGDB_PAGE} 의 Drive 링크에서 audio_DI를 받아 {out}에 두세요.")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    a = ap.parse_args(argv)
    print("NAM 표준 입력:", get_nam_input(a.data))
    print("EGDB DI:", get_egdb(a.data), "개")


if __name__ == "__main__":
    main()

"""Export-only tooling: never imports the source project or writes to it."""
import ctypes
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'original_source'
RUN = 'mt_m0_avstd_s1111_v1'

if os.name == 'nt':
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
    kernel.CreateFileW.restype = w.HANDLE
    kernel.SetFileTime.argtypes = [w.HANDLE, ctypes.POINTER(w.FILETIME), ctypes.POINTER(w.FILETIME), ctypes.POINTER(w.FILETIME)]
    kernel.SetFileTime.restype = w.BOOL
    kernel.ReadFile.argtypes = [w.HANDLE, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD), w.LPVOID]
    kernel.ReadFile.restype = w.BOOL
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.GetFileInformationByHandleEx.argtypes = [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD]
    kernel.GetFileInformationByHandleEx.restype = w.BOOL

class DirectoryInfo(ctypes.Structure):
    _fields_ = [('NextEntryOffset', w.DWORD), ('FileIndex', w.DWORD),
                ('CreationTime', ctypes.c_longlong), ('LastAccessTime', ctypes.c_longlong),
                ('LastWriteTime', ctypes.c_longlong), ('ChangeTime', ctypes.c_longlong),
                ('EndOfFile', ctypes.c_longlong), ('AllocationSize', ctypes.c_longlong),
                ('FileAttributes', w.DWORD), ('FileNameLength', w.DWORD),
                ('EaSize', w.DWORD), ('ShortNameLength', ctypes.c_byte),
                ('ShortName', w.WCHAR * 12), ('FileId', ctypes.c_longlong),
                ('FileName', w.WCHAR * 1)]

def safe_entries(directory):
    """Enumerate using the SAME no-atime-update directory handle (Windows)."""
    handle = kernel.CreateFileW(str(directory), 1 | 0x80 | 0x100, 7, None, 3, 0x02000000, None)
    if handle == w.HANDLE(-1).value:
        raise OSError(f'Cannot enumerate with atime suppression: {directory}; {ctypes.WinError(ctypes.get_last_error())}')
    try:
        sentinel = w.FILETIME(0xFFFFFFFF, 0xFFFFFFFF)
        if not kernel.SetFileTime(handle, None, ctypes.byref(sentinel), None):
            raise ctypes.WinError(ctypes.get_last_error())
        buf = ctypes.create_string_buffer(65536)
        restart = True
        while True:
            if not kernel.GetFileInformationByHandleEx(handle, 11 if restart else 10, buf, len(buf)):
                error = ctypes.get_last_error()
                if error == 18:
                    break
                raise ctypes.WinError(error)
            restart = False
            offset = 0
            while True:
                entry = DirectoryInfo.from_buffer(buf, offset)
                name = ctypes.wstring_at(ctypes.addressof(buf) + offset + DirectoryInfo.FileName.offset,
                                         entry.FileNameLength // 2)
                if name not in ('.', '..'):
                    yield directory / name, bool(entry.FileAttributes & 0x10)
                if entry.NextEntryOffset == 0:
                    break
                offset += entry.NextEntryOffset
    finally:
        kernel.CloseHandle(handle)

def safe_snapshot(fallback=None, blocked=None):
    result = {'.': dict(kind='directory', **metadata(ORIGINAL))}
    pending = [ORIGINAL]
    while pending:
        directory = pending.pop()
        try:
            entries = list(safe_entries(directory))
        except OSError as error:
            if fallback is None:
                raise
            if blocked is not None:
                blocked.append({'path': str(directory), 'reason': str(error)})
            for rel, prior in fallback.items():
                candidate = ORIGINAL / rel
                if candidate.is_relative_to(directory) and candidate.exists():
                    result[rel] = dict(kind=prior['kind'], **metadata(candidate))
            continue
        for path, is_directory in entries:
            result[path.relative_to(ORIGINAL).as_posix()] = dict(
                kind='directory' if is_directory else 'file', **metadata(path))
            if is_directory:
                pending.append(path)
    return result

def chunks(path):
    """SetFileTime(-1) suppresses atime updates on this handle, not a date edit.

    https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-setfiletime
    No source data, timestamp value, or global filesystem setting is written.
    """
    if os.name != 'nt':
        with open(path, 'rb') as stream:
            while block := stream.read(4 * 1024 * 1024):
                yield block
        return
    handle = kernel.CreateFileW(str(path), 0x80000000 | 0x100, 7, None, 3, 0x08000000, None)
    if handle == w.HANDLE(-1).value:
        raise OSError(f'Cannot open with atime suppression: {path}; {ctypes.WinError(ctypes.get_last_error())}')
    try:
        sentinel = w.FILETIME(0xFFFFFFFF, 0xFFFFFFFF)
        if not kernel.SetFileTime(handle, None, ctypes.byref(sentinel), None):
            raise ctypes.WinError(ctypes.get_last_error())
        buf = ctypes.create_string_buffer(max(1, min(path.stat().st_size, 4 * 1024 * 1024)))
        count = w.DWORD()
        while True:
            if not kernel.ReadFile(handle, buf, len(buf), ctypes.byref(count), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if not count.value:
                break
            yield ctypes.string_at(buf, count.value)
    finally:
        kernel.CloseHandle(handle)

def digest(path):
    sha = hashlib.sha256()
    for block in chunks(path):
        sha.update(block)
    return sha.hexdigest()

def metadata(path):
    s = path.stat()
    return dict(size_bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                atime_ns=s.st_atime_ns, creation_ns=s.st_birthtime_ns,
                attributes=s.st_file_attributes)

def save(name, value):
    dest = ROOT / 'q3_adapter' / 'audit' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def snapshot():
    result = {}
    for path in [ORIGINAL, *sorted(ORIGINAL.rglob('*'))]:
        rel = path.relative_to(ORIGINAL).as_posix()
        result[rel] = dict(kind='directory' if path.is_dir() else 'file', **metadata(path))
    return result

def copy_one(src, dest, purpose, required=True, resume_existing=False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    before = metadata(src)
    original_sha = digest(src)
    if dest.exists() and not resume_existing:
        raise FileExistsError(dest)
    if not dest.exists():
        with dest.open('xb') as stream:
            for block in chunks(src):
                stream.write(block)
    copied_sha = digest(dest)
    after_sha = digest(src)
    assert original_sha == copied_sha == after_sha, str(src)
    after = metadata(src)
    assert before == after, (str(src), before, after)
    os.utime(dest, ns=(before['atime_ns'], before['mtime_ns']))
    return dict(relative_path=dest.relative_to(ROOT).as_posix(), original_absolute_path=str(src),
                size_bytes=before['size_bytes'], sha256=original_sha,
                source_sha256_before=original_sha, source_sha256_after=after_sha,
                copy_sha256=copied_sha, purpose=purpose, required_for_runtime=required,
                original_unmodified_copy=True, generated_this_task=False,
                original_metadata_before=before, original_metadata_after=after)

def export():
    assert not ROOT.is_relative_to(ORIGINAL)
    before = snapshot()
    save('source_inventory_before.json', before)
    print('Source inventory:', len(before), 'entries;', sum(x['size_bytes'] for x in before.values() if x['kind'] == 'file'), 'bytes', flush=True)
    provenance_path = ORIGINAL / 'outputs' / RUN / 'provenance.json'
    provenance = json.loads(b''.join(chunks(provenance_path)))
    paths = set(provenance['code_sha256'])
    # Hash checks are the minimal strict closure; package init files preserve imports.
    for package in ['baseline', 'robust', 'standardized', 'multitask', 'dataset_construction']:
        paths.update(p.relative_to(ORIGINAL).as_posix() for p in (ORIGINAL/package).glob('*.py'))
    records = []
    for rel in sorted(paths):
        records.append(copy_one(ORIGINAL / rel, ROOT / 'original_source' / rel,
                                'Original implementation / strict checkpoint code-hash closure'))
    for path in sorted((ORIGINAL / 'pretrained' / 'bert-base-uncased').iterdir()):
        if path.is_file():
            records.append(copy_one(path, ROOT / 'original_source' / path.relative_to(ORIGINAL),
                                    'Local BERT construction/tokenizer/base weights and provenance',
                                    path.suffix not in ['.md'] and path.name != 'LICENSE'))
    for name in ['best.pt', 'config.json', 'preprocessing.json', 'provenance.json', 'evaluation.json', 'history.json', 'valid__clean.csv', 'test__clean.csv']:
        candidates = [ORIGINAL / 'outputs' / RUN / name, ORIGINAL / 'outputs' / RUN / 'details' / name]
        src = next((p for p in candidates if p.is_file()), None)
        if src is None:
            raise FileNotFoundError(name)
        records.append(copy_one(src, ROOT / 'checkpoint' / name, 'Formal M0 checkpoint/run record', name == 'best.pt'))
    for rel in ['requirements.txt', '_dev/records/environment/requirements-lock.txt']:
        if (ORIGINAL/rel).is_file():
            records.append(copy_one(ORIGINAL/rel, ROOT/'original_source'/rel, 'Original environment reference', False))
    save('copy_records.json', records)
    save('source_inventory_after_copy.json', snapshot())
    print('Copied and triple-hash-verified:', len(records), 'files', flush=True)

if __name__ == '__main__':
    export()

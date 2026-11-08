"""Fetch pinned, unmodified Qwen weights; verify the publisher's SHA-256.

Hugging Face supplies the immutable revision and expected file checksums.
The same Qwen-published files are downloaded from ModelScope
in resumable ranges to avoid the slow large-file route on this connection.
No API key or project configuration is read.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen3-Embedding-0.6B'
REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
DESTINATION = ROOT / '.cache' / 'models' / 'Qwen3-Embedding-0.6B'
CHUNK_SIZE = 16 * 1024 * 1024


def request(url, headers=None):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (OSError, urllib.error.URLError):
            if attempt == 3:
                raise
        print(f'Download connection interrupted; retry {attempt + 1}/3 with TLS verification enabled.', flush=True)
        time.sleep(2 ** attempt)


def read_bytes(url):
    for attempt in range(3):
        try:
            with request(url) as response:
                return response.read()
        except (OSError, urllib.error.URLError, http.client.IncompleteRead):
            if attempt == 2:
                raise
            print(f'File transfer interrupted; retry {attempt + 1}/2.', flush=True)
            time.sleep(2 ** attempt)


def download_weight(item, destination):
    expected = item['lfs']['sha256']
    if destination.exists() and destination.stat().st_size == item['size']:
        with destination.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() == expected:
                print('Verified cached model weights.', flush=True)
                return
    chunks = DESTINATION / '_download_chunks'
    chunks.mkdir(exist_ok=True)

    def get_chunk(index):
        start = index * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, item['size']) - 1
        path = chunks / f'{index:04d}.part'
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        url = f'https://www.modelscope.cn/models/{MODEL}/resolve/master/model.safetensors?chunk={index}'
        for attempt in range(3):
            try:
                with request(url, {'Range': f'bytes={start}-{end}'}) as response:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                        raise ValueError('Server did not honor requested byte range')
                    with path.open('wb') as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                if path.stat().st_size != end - start + 1:
                    raise ValueError('Incomplete download chunk')
                return path
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    count = (item['size'] + CHUNK_SIZE - 1) // CHUNK_SIZE
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(get_chunk, index) for index in range(count)]
        for complete, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f'Embedding weights: {complete}/{count} chunks complete; elapsed {time.monotonic() - started:.0f}s', flush=True)
    assembled = destination.with_suffix('.assembling')
    with assembled.open('wb') as stream:
        for index in range(count):
            with (chunks / f'{index:04d}.part').open('rb') as chunk:
                shutil.copyfileobj(chunk, stream, length=4 * 1024 * 1024)
    with assembled.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('Weights differ from pinned publisher SHA-256; refusing to use them')
    assembled.replace(destination)
    # Leave chunk files in the ignored cache for safe recovery; no recursive delete.


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(DESTINATION).free < 4_000_000_000:
        raise RuntimeError('At least 4 GB free space is needed for download and assembly')
    metadata = json.loads(read_bytes(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true'))
    if metadata['sha'] != REVISION:
        raise ValueError('Unexpected model metadata revision')
    files = [item for item in metadata['siblings']
             if item['rfilename'].endswith(('.json', '.txt', '.safetensors'))
             and not item['rfilename'].startswith(('onnx/', 'openvino/'))]
    for item in files:
        destination = DESTINATION / item['rfilename']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if item['rfilename'] == 'model.safetensors':
            download_weight(item, destination)
        elif not destination.exists() or destination.stat().st_size != item['size']:
            print('Downloading configuration:', item['rfilename'], flush=True)
            data = read_bytes(f'https://www.modelscope.cn/models/{MODEL}/resolve/master/{item["rfilename"]}')
            if len(data) != item['size']:
                raise ValueError('Configuration size mismatch')
            if item.get('lfs'):
                if hashlib.sha256(data).hexdigest() != item['lfs']['sha256']:
                    raise ValueError('LFS configuration differs from the pinned revision')
            else:
                expected_blob = item.get('blobId')
                git_blob = b'blob ' + str(len(data)).encode() + b'\0' + data
                if expected_blob and hashlib.sha1(git_blob).hexdigest() != expected_blob:
                    raise ValueError('Configuration differs from the pinned revision')
            destination.write_bytes(data)
    provenance = {'repository': MODEL, 'revision': REVISION, 'files': files,
                  'weights_download_source': 'Qwen official ModelScope mirror; verified against Hugging Face LFS SHA-256'}
    path = ROOT / 'reproduction' / 'data' / 'embedding_provenance.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print('Original embedding model is ready. SHA-256 verified. API calls: 0.', flush=True)


if __name__ == '__main__':
    main()

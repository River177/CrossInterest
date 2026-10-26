"""Download only committed formal checkpoints, safely reusing verified bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import tempfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceNotFoundError
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--prefix',required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--identity',required=True);p.add_argument('--steps',type=int,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if not re.fullmatch(r'projects/sft4safety/amlt-results/[A-Za-z0-9.-]+',a.prefix):raise ValueError('Invalid prefix')
    if not re.fullmatch(r'\d{8}T\d{6}Z-[a-f0-9]{8}',a.attempt):raise ValueError('Invalid attempt')
    if any(step not in range(25,676,25) for step in a.steps):raise ValueError('Invalid formal checkpoint')
    c=BlobServiceClient('https://csnewsandfeeds4150361735.blob.core.windows.net',credential=AzureCliCredential(),
        connection_timeout=10,read_timeout=25,retry_total=2).get_container_client('amulet')
    report={'downloaded':[],'not_committed':[]}
    for step in a.steps:
        prefix=f'{a.prefix}/attempts/{a.attempt}/checkpoint-{step}/'
        try:body=c.download_blob(prefix+'checkpoint_manifest.json').readall()
        except ResourceNotFoundError:
            report['not_committed'].append(step);continue
        m=json.loads(body)
        if m['identity']!=a.identity:raise ValueError('Checkpoint identity differs')
        dest=a.output_dir/f'checkpoint-{step}';dest.mkdir(parents=True,exist_ok=True)
        manifest=dest/'checkpoint_manifest.json'
        if manifest.exists() and manifest.read_bytes()!=body:raise ValueError('Existing checkpoint manifest differs')
        if not manifest.exists():manifest.write_bytes(body)
        def fetch(item):
            name,expected=item;relative=PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid manifest path')
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target)!=expected:raise ValueError(f'Existing file differs: {name}')
                return target.stat().st_size
            temporary=None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.download-',delete=False) as stream:
                    temporary=Path(stream.name);c.download_blob(prefix+name,max_concurrency=2).readinto(stream)
                if sha(temporary)!=expected:raise ValueError(f'Download hash differs: {name}')
                os.replace(temporary,target)
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)
            return target.stat().st_size
        with ThreadPoolExecutor(max_workers=8) as pool:sizes=list(pool.map(fetch,m['files'].items()))
        receipt={'step':step,'identity':a.identity,'manifest_sha256':hashlib.sha256(body).hexdigest(),'files':len(sizes),'bytes':sum(sizes)}
        (a.output_dir/f'checkpoint-{step}-download.json').write_text(json.dumps(receipt,indent=2)+'\n')
        report['downloaded'].append(receipt)
    print(json.dumps(report))


if __name__=='__main__':main()

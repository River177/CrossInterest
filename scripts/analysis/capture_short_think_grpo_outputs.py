"""Read scoped GRPO output progress and download a committed attempt archive.

Authentication stays in memory. Complete attempts are selected explicitly; no
cross-attempt merge or mutable-root result fallback is allowed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(content,indent=2)+'\n')


def unpack(archive,destination):
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive member')
        for member in members:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile():
                raise ValueError('Unsafe or non-file archive member')
            target=destination.joinpath(*path.parts);target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,target.open('wb') as out:
                while block:=src.read(8*1024*1024):out.write(block)


def main():
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobServiceClient
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--download-to',type=Path)
    parser.add_argument('--max-package-bytes',type=int,default=2_000_000_000)
    args=parser.parse_args()
    if not 0<args.max_package_bytes<=64_000_000_000:raise ValueError('Invalid explicit package bound')
    if not re.fullmatch(r'projects/sft4safety/amlt-results/[A-Za-z0-9.-]+/?',args.prefix):raise ValueError('Invalid results prefix')
    if not re.fullmatch(r'\d{8}T\d{6}Z-[a-f0-9]{8}',args.attempt):raise ValueError('Invalid attempt')
    root=args.output_dir;root.mkdir(parents=True,exist_ok=False)
    prefix=args.prefix.rstrip('/')+'/'
    container=BlobServiceClient('https://csnewsandfeeds4150361735.blob.core.windows.net',
        credential=AzureCliCredential(),connection_timeout=10,read_timeout=25,retry_total=2).get_container_client('amulet')
    items=[{'name':b.name[len(prefix):],'size':b.size,'last_modified':b.last_modified.isoformat()}
           for b in container.list_blobs(name_starts_with=prefix)]
    write(root/'blob_inventory.json',items)
    attempt_prefix='attempts/'+args.attempt+'/'
    groups=[b for b in items if b['name'].startswith(attempt_prefix)
            and re.fullmatch(r'shard-[0-3]/group-\d{3}\.parquet',b['name'][len(attempt_prefix):]) and b['size']>0]
    receipt={'observed_at_utc':datetime.now(timezone.utc).isoformat(),'prefix':prefix,'attempt':args.attempt,
        'persisted_group_files':len(groups),'groups_by_shard':dict(Counter(b['name'].split('/')[-2] for b in groups)),
        'file_count':len(items),'downloaded':[],'complete':False}
    def fetch(name):
        body=container.download_blob(prefix+name).readall()
        target=root/'readback'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
        receipt['downloaded'].append({'name':name,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
        return json.loads(body)
    complete_name=f'completed-{args.attempt}.json'
    for item in items:
        name=item['name']
        if name.startswith(attempt_prefix) and name.endswith(('complete.json','metrics.json','package.json','identity.json')):
            if item['size']>2_000_000:raise ValueError('Unexpected receipt size')
            fetch(name)
    if any(b['name']==complete_name and b['size']>0 for b in items):
        completed=fetch(complete_name);package=fetch(attempt_prefix+'package.json')
        if completed!=package or package['attempt']!=args.attempt:raise ValueError('Completion binding differs')
        receipt['complete']=True
        if args.download_to:
            dest=args.download_to
            if dest.exists():raise FileExistsError(dest)
            dest.mkdir(parents=True)
            archive=dest/'results.tar.gz'
            if not 0<package['bytes']<=args.max_package_bytes:raise ValueError('Unexpected package size')
            with archive.open('wb') as stream:
                container.download_blob(prefix+attempt_prefix+'results.tar.gz',max_concurrency=4).readinto(stream)
            if archive.stat().st_size!=package['bytes'] or sha(archive)!=package['sha256']:raise ValueError('Archive integrity differs')
            write(dest/'package.json',package)
            unpack(archive,dest/'result')
            manifest=dest/'result/artifact_manifest.json'
            if sha(manifest)!=package['manifest_sha256']:raise ValueError('Manifest binding differs')
            receipt['download']={'directory':str(dest.resolve()),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size}
    write(root/'receipt.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()

"""Upload and independently read back H5 artifacts directly from GCR to Azure.

Signed URLs are passed over SSH stdin, never printed or stored. Run the local
mode using an Azure-authenticated Python; only compact receipts reach the Mac.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ACCOUNT = 'csnewsandfeeds4150361735'
PREFIX = 'projects/sft4safety/manual-results/history-h5-original-profile-20260916-v1'
REMOTE_HOST_ROOT = '/home/v-wuhao1/experiments/history-h5-repair-20260916'


def remote():
    import requests
    payload = json.load(sys.stdin)
    def transfer(pair):
        name, item = pair
        try:
            path = Path(item['path'])
            assert str(path).startswith('/task/')
            with path.open('rb') as stream:
                response = requests.put(item['url'], data=stream, headers={
                    'x-ms-blob-type':'BlockBlob','If-None-Match':'*'}, timeout=(15,180))
            if response.status_code not in (201,412):
                raise RuntimeError('Upload HTTP ' + str(response.status_code))
            digest = hashlib.sha256()
            size = 0
            target = Path('/task/azure-readback') / name
            target.parent.mkdir(parents=True,exist_ok=True)
            with requests.get(item['url'],stream=True,timeout=(15,180)) as result:
                if result.status_code != 200:
                    raise RuntimeError('Readback HTTP '+str(result.status_code))
                with target.open('wb') as stream:
                    for chunk in result.iter_content(1048576):
                        digest.update(chunk); size += len(chunk);stream.write(chunk)
            assert digest.hexdigest() == item['sha256'] and size == item['bytes']
            return name, {'sha256':digest.hexdigest(),'bytes':size,'uploaded_and_readback':True}
        except Exception as exc:
            # Requests exceptions may include signed URLs; do not serialize them.
            raise RuntimeError('Transfer failed for '+name+' ('+type(exc).__name__+')') from None
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = dict(pool.map(transfer,payload['files'].items()))
    target = Path('/task/azure-readback')
    manifest = json.loads((target / 'artifact_manifest.json').read_text())
    expected = {k:v['sha256'] for k,v in manifest['files'].items()}
    expected['artifact_manifest.json'] = payload['files']['artifact_manifest.json']['sha256']
    expected['complete.json'] = payload['files']['complete.json']['sha256']
    found = set()
    with tarfile.open(target / 'publication.tar.gz','r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            assert member.name.startswith('publication/')
            relative = member.name.removeprefix('publication/')
            assert relative in expected
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda:stream.read(1048576),b''):
                    digest.update(chunk)
            assert digest.hexdigest() == expected[relative]
            assert relative not in found
            found.add(relative)
    assert found == set(expected)
    receipt = {'status':'azure_persistent_upload_full_archive_readback_verified',
        'prefix':PREFIX,'archive_member_files_verified':len(found),'files':verified,
        'verified_at_utc':datetime.now(timezone.utc).isoformat()}
    (Path('/task') / 'azure_publication_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


def local(output):
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
    result = subprocess.run(['ssh','-o','BatchMode=yes','gcr-a100','cat '+REMOTE_HOST_ROOT+'/h5_transfer_inventory.json'],capture_output=True,check=True)
    inventory = json.loads(result.stdout)
    client = BlobServiceClient('https://'+ACCOUNT+'.blob.core.windows.net',credential=AzureCliCredential())
    now = datetime.now(timezone.utc)
    delegation = client.get_user_delegation_key(now-timedelta(minutes=5),now+timedelta(hours=1))
    payload = {'files':{}}
    for name,item in inventory['files'].items():
        blob = PREFIX+'/'+name
        sas = generate_blob_sas(ACCOUNT,'amulet',blob,user_delegation_key=delegation,
            permission=BlobSasPermissions(read=True,write=True,create=True),
            expiry=now+timedelta(minutes=45),start=now-timedelta(minutes=5))
        payload['files'][name] = {**item,'url':'https://'+ACCOUNT+'.blob.core.windows.net/amulet/'+blob+'?'+sas}
    command=['ssh','-o','BatchMode=yes','gcr-a100',
        'docker exec -i ci-h5-repair-0916 python /task/code/persist_history_h5_publication.py --remote']
    result=subprocess.run(command,input=json.dumps(payload).encode(),capture_output=True)
    if result.returncode:
        # Keep potential network details from unexpected stderr out of logs.
        raise RuntimeError('Remote persistent publication failed; inspect sanitized remote receipt')
    receipt=json.loads(result.stdout)
    assert receipt['status']=='azure_persistent_upload_full_archive_readback_verified'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    # Persist the compact verification receipt with OAuth and verify its bytes.
    b=client.get_blob_client('amulet',PREFIX+'/azure_publication_receipt.json')
    data=output.read_bytes()
    if not b.exists():
        b.upload_blob(data,overwrite=False)
    assert b.download_blob().readall()==data
    print('AZURE_PERSISTENT_PUBLICATION_VERIFIED',receipt['archive_member_files_verified'],'files')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.remote:
        remote()
    else:
        if args.output is None:
            parser.error('--output is required locally')
        local(args.output)

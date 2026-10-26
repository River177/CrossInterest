"""Bounded, credential-redacted native logs for an explicitly named pilot backend."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import requests
from azure.identity import AzureCliCredential


WORKSPACE='subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895/resourceGroups/rg-cs-newsandfeeds-singularity/providers/Microsoft.MachineLearningServices/workspaces/CS-NewsAndFeeds-Singularity'


def redact(text):
    text=re.sub(r'eyJ[A-Za-z0-9_.-]{50,}','[REDACTED_JWT]',text)
    text=re.sub(r'(?i)bearer\s+[A-Za-z0-9._~-]+','Bearer [REDACTED]',text)
    text=re.sub(r'(?i)([?&](?:sig|token|access_token|sv)=)[^&\s\"\']+',r'\1[REDACTED]',text)
    return re.sub(r'https?://[^\s"<>]+',lambda m:m.group().split('?')[0]+'?[REDACTED_QUERY]'
        if re.search(r'[?&](?:sig|sv|token|access_token)=',m.group(),re.I) else m.group(),text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+',a.backend):raise ValueError('Invalid backend ID')
    a.output.mkdir(parents=True,exist_ok=True)
    token=AzureCliCredential(process_timeout=30).get_token('https://management.azure.com/.default').token
    headers={'Authorization':'Bearer '+token};base='https://ml.azure.com/api/eastus'
    def api(url,**kw):
        r=requests.get(url,headers=headers,timeout=(15,40),**kw)
        if r.status_code!=200:raise RuntimeError(f'Azure API HTTP{r.status_code}')
        return r.json()
    details=api(f'{base}/history/v1.0/{WORKSPACE}/runs/{a.backend}/details')
    logs=list((details.get('logFiles') or {}).keys())
    names=[n for n in logs if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log\.txt',n)]
    names += [n for n in logs if n=='logs/azureml/virtualcluster_activity_log.txt']
    result={'backend':a.backend,'observed_utc':datetime.now(timezone.utc).isoformat(),
        'native_status':details.get('status'),'start_time':details.get('startTimeUtc'),'logs':[]}
    for i,name in enumerate(names):
        info=api(f'{base}/artifact/v2.0/{WORKSPACE}/artifacts/ExperimentRun/dcid.{a.backend}/contentinfo',params={'path':name})
        uri=info['contentUri'];meta=requests.head(uri,timeout=(15,40))
        if meta.status_code!=200:raise RuntimeError(f'Log HEAD HTTP{meta.status_code}')
        total=int(meta.headers.get('Content-Length','0'));offset=max(0,total-131072)
        if not total:continue
        r=requests.get(uri,headers={'Range':f'bytes={offset}-'},timeout=(15,40))
        if r.status_code not in (200,206):raise RuntimeError(f'Log range HTTP{r.status_code}')
        text=redact(r.content.decode(errors='replace')).replace('\r','\n')
        out=a.output/f'log-{i}.txt';out.write_text(text)
        result['logs'].append({'name':name,'file':str(out),'remote_bytes':total,'offset':offset,
            'last_modified':meta.headers.get('Last-Modified'),'redacted_sha256':hashlib.sha256(text.encode()).hexdigest()})
        print(name+'\n'+'\n'.join(text.splitlines()[-45:]))
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'native_status':result['native_status'],'streams':len(result['logs'])}))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(redact(type(error).__name__+': '+str(error)),file=sys.stderr)
        sys.exit(1)

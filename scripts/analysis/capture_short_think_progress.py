"""Read native status and every bounded stdout branch without changing jobs."""
import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re

import requests
from azure.identity import AzureCliCredential

WORKSPACE='subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895/resourceGroups/rg-cs-newsandfeeds-singularity/providers/Microsoft.MachineLearningServices/workspaces/CS-NewsAndFeeds-Singularity'


def redact(text):
    text=re.sub(r'eyJ[A-Za-z0-9_.-]{50,}','[REDACTED_JWT]',text)
    text=re.sub(r'(?i)(\b(?:accessToken|access_token|authorization|AZUREML_RUN_TOKEN|WANDB_API_KEY|HF_TOKEN)\b["\s]*[:=]\s*)[^\n]+',r'\1[REDACTED]',text)
    return re.sub(r'https?://[^\s"<>]+',lambda m:m.group().split('?')[0]+'?[REDACTED_QUERY]' if re.search(r'[?&](?:sig|token|access_token|sv)=',m.group(),re.I) else m.group(),text)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--backend',required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.backend):raise ValueError('Invalid backend')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    token=AzureCliCredential().get_token('https://management.azure.com/.default').token
    headers={'Authorization':'Bearer '+token};base='https://ml.azure.com/api/eastus'
    def metadata(url,**kwargs):
        r=requests.get(url,headers=headers,timeout=(10,25),**kwargs)
        if r.status_code!=200:raise RuntimeError(f'MetadataHTTP{r.status_code}')
        return r.json()
    receipt={'backend':args.backend,'observed_at_utc':datetime.now(timezone.utc).isoformat(),'logs':{}}
    try:
        detail=metadata(f'{base}/history/v1.0/{WORKSPACE}/runs/{args.backend}/details')
        receipt.update(native_status=detail.get('status'),start_time_utc=detail.get('startTimeUtc'),end_time_utc=detail.get('endTimeUtc'))
        names=list(detail.get('logFiles') or {})
        selected=[n for n in names if re.fullmatch(r'user_logs/(?:retry_\d+/)?std_log\.txt',n) or n=='logs/azureml/virtualcluster_activity_log.txt']
        receipt['available_log_names']=names
        for name in selected:
            info=metadata(f'{base}/artifact/v2.0/{WORKSPACE}/artifacts/ExperimentRun/dcid.{args.backend}/contentinfo',params={'path':name})
            uri=info['contentUri']
            if not uri.startswith('https://'):raise ValueError('Invalid contentURI')
            head=requests.head(uri,timeout=(10,25))
            if head.status_code!=200:raise RuntimeError(f'ContentHTTP{head.status_code}')
            size=int(head.headers.get('Content-Length',0));offset=max(0,size-65536)
            if not size:receipt['logs'][name]={'remote_bytes':0};continue
            res=requests.get(uri,headers={'Range':f'bytes={offset}-'},timeout=(10,25))
            if res.status_code not in (200,206):raise RuntimeError(f'RangeHTTP{res.status_code}')
            text=redact(res.content.decode(errors='replace')).replace('\r','\n')
            path=out/name.replace('/','__');path.write_text(text)
            receipt['logs'][name]={'local_file':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'remote_bytes':size,'offset':offset,'last_modified':res.headers.get('Last-Modified'),
                'markers':[line[:900] for line in text.splitlines() if any(s in line for s in ['SHORT_THINK','GRPO_','CHECKPOINT_','ScheduleNotPossible','Suspended','Requeued','Traceback','Error:'])][-12:]}
    except Exception as error:
        receipt['observation_error_type']=type(error).__name__
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()

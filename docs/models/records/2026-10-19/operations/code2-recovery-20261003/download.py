import os,json
from pathlib import Path
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
revision='15852e8c16360a2fea060d615a32b45270f8a8fc'
root=Path('/root/autodl-tmp/lrr/ParamAgent/models/Qwen3.5-2B')
print('DOWNLOAD_START',revision,flush=True)
snapshot_download('Qwen/Qwen3.5-2B',revision=revision,local_dir=root,max_workers=2,allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','LICENSE','README.md'])
(root/'download_provenance.json').write_text(json.dumps({'repo_id':'Qwen/Qwen3.5-2B','revision':revision,'endpoint':'https://hf-mirror.com'},indent=2))
print('DOWNLOAD_COMPLETE',root,flush=True)

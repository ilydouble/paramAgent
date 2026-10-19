import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
import gc,json,hashlib,shutil,datetime
from pathlib import Path
import torch,transformers,peft
from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer
from peft import PeftModel
from safetensors import safe_open
root=Path('/root/autodl-tmp/lrr/ParamAgent')
base=root/'models/Qwen3.5-2B'
adapter=root/'lora-qwen3.5-2b/lora-qwen3.5-2b-qa'
out=root/'models/Qwen3.5-2B-qa-merged'
staging=out.with_name(out.name+'.partial-20261003')
assert (base/'download_provenance.json').is_file(),'Base download not complete'
assert not out.exists() and not staging.exists(),'Refuse to overwrite output'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
assert sha(base/'model.safetensors-00001-of-00001.safetensors')=='aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1','Base SHA256 mismatch'
print('BASE_SHA256_VERIFIED',flush=True)
torch.set_num_threads(8)
print('LOAD_BASE_CPU',flush=True)
model=Qwen3_5ForConditionalGeneration.from_pretrained(base,dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='eager',local_files_only=True)
with safe_open(adapter/'adapter_model.safetensors',framework='pt',device='cpu') as f:
 for k in f.keys():
  name=k.removeprefix('base_model.model.').split('.lora_')[0]
  layer=model.get_submodule(name)
  shape=f.get_slice(k).get_shape()
  assert (shape[1]==layer.in_features if '.lora_A.' in k else shape[0]==layer.out_features),(k,shape,layer.weight.shape)
print('ADAPTER_SHAPES_MATCH',flush=True)
model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True)
layers=[(n,m) for n,m in model.named_modules() if hasattr(m,'lora_A') and 'default' in m.lora_A]
assert len(layers)==114,len(layers)
name,layer=layers[0]
expected=layer.get_base_layer().weight.detach().clone(); expected.add_(layer.get_delta_weight('default').detach().to(expected.dtype))
print('MERGE_LAYERS',len(layers),flush=True)
merged=model.merge_and_unload(safe_merge=True)
assert not any('lora_' in n for n,_ in merged.named_parameters())
layername=name.removeprefix('base_model.model.')
assert torch.equal(merged.get_submodule(layername).weight,expected),'Merged tensor differs from expected delta'
staging.mkdir()
merged.save_pretrained(staging,safe_serialization=True,max_shard_size='5GB')
tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
tok.save_pretrained(staging)
for n in ['preprocessor_config.json','video_preprocessor_config.json','chat_template.jinja']:
 if (base/n).exists():shutil.copy2(base/n,staging/n)
del model,merged,layers,layer,expected;gc.collect()
print('RELOAD_MERGED_CPU',flush=True)
check=Qwen3_5ForConditionalGeneration.from_pretrained(staging,dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='eager',local_files_only=True)
messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'What is the capital of France? Answer briefly.'}]
prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
inputs=tok(prompt,return_tensors='pt')
check.eval()
with torch.inference_mode():
 logits=check(**inputs,use_cache=False).logits
 assert torch.isfinite(logits).all(),'Nonfinite logits'
 result=check.generate(**inputs,max_new_tokens=12,do_sample=False)
text=tok.decode(result[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
assert text.strip(),'Empty smoke response'
print('SMOKE_RESPONSE',repr(text),flush=True)
files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in staging.glob('*.safetensors')}
report={'status':'verified','kind':'SFT_only','base':json.loads((base/'download_provenance.json').read_text()),'historical_base_revision_verified':False,'adapter':str(adapter),'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'merged_layers':114,'dtype':'bfloat16','device':'cpu','versions':{'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},'verification':{'adapter_shapes_match':True,'selected_delta_match':True,'reload_success':True,'finite_logits':True,'smoke_response':text},'weights':files,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(staging/'merge_provenance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
staging.rename(out)
print('MERGE_VERIFIED',out,flush=True)

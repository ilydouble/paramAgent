from pathlib import Path
import sys,json,hashlib,torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoProcessor
from peft import PeftModel
r=Path('/root/autodl-tmp/lrr/ParamAgent')
w=Path('/root/autodl-tmp/lrr/code2-dpo-run-20261003')
a=r/'lora-qwen3.5-2b-dpo/lora-qwen3.5-2b-code2-dpo-20261003'
b=r/'models/Qwen3.5-2B-code-merged2'
assert (w/'train.exit').read_text().strip()=='0'
c=json.loads((a/'adapter_config.json').read_text())
assert Path(c['base_model_name_or_path']).resolve()==b.resolve(),c['base_model_name_or_path']
st=load_file(str(a/'adapter_model.safetensors'))
assert all(torch.isfinite(t).all().item() for t in st.values())
assert any('lora_B' in k and torch.count_nonzero(t).item()>0 for k,t in st.items())
sys.path.insert(0,str(r/'code'))
from LoRA_Qwen35_Code_4090 import SYSTEM_PROMPT
p=AutoProcessor.from_pretrained(str(b)); tok=getattr(p,'tokenizer',p)
m=AutoModelForCausalLM.from_pretrained(str(b),dtype=torch.bfloat16,device_map={'':0})
model_class=type(m).__name__
m=PeftModel.from_pretrained(m,str(a)).eval()
messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':'FUNC_SIGNATURE:'+chr(10)+'def find_max(numbers: list[int]) -> int:'}]
prompt=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
x=tok(prompt,return_tensors='pt').to('cuda')
with torch.inference_mode():
 y=m(**x).logits[:,-1,:].float()
 assert torch.isfinite(y).all().item()
 with m.disable_adapter(): z=m(**x).logits[:,-1,:].float()
 delta=(y-z).abs().max().item();assert delta>0,delta
 out=m.generate(**x,max_new_tokens=128,do_sample=False,pad_token_id=tok.eos_token_id)
 completion=tok.decode(out[0,x['input_ids'].shape[1]:],skip_special_tokens=True)
assert completion.strip()
report={'status':'verified','base':str(b),'adapter':str(a),'model_class':model_class,'adapter_sha256':hashlib.sha256((a/'adapter_model.safetensors').read_bytes()).hexdigest(),'tensor_count':len(st),'all_weights_finite':True,'adapter_changes_logits_max_abs':delta,'smoke_generation':completion,'note':'Load and generation smoke check; not a quality evaluation.'}
(w/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))

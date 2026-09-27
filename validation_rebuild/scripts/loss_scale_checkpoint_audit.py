"""No optimization: deterministic forward-only loss audit of frozen M5 (paper M4)."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch,yaml
R=Path(__file__).resolve().parents[2];O=R/'validation_rebuild'
sys.path.insert(0,str(R/'data_driven_alpha_beta/code_snapshot'))
from scripts.train.train import build_datasets,move_batch,set_seed
from scripts.train.model import build_model_from_config
from scripts.train.losses import build_total_loss
from torch.utils.data import DataLoader,Subset
set_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
cp=torch.load(R/'data_driven_alpha_beta/M5/training/checkpoints/best.pt',map_location='cpu',weights_only=False);c=cp['config'];ds,_,_=build_datasets(c)
idx=np.linspace(0,len(ds)-1,192,dtype=int);np.savetxt(O/'audit/loss_audit_train_indices.csv',idx,fmt='%d',header='train_tile_index',comments='')
model=build_model_from_config(int(cp['model_state_dict']['stem.block.0.weight'].shape[1]),c['model']);model.load_state_dict(cp['model_state_dict']);model.to('cuda').eval();rows=[]
with torch.no_grad():
 for i,b in enumerate(DataLoader(Subset(ds,idx.tolist()),batch_size=4,shuffle=False,num_workers=0)):
  b=move_batch(b,torch.device('cuda'));p=model(b['inputs'],coarse_context=b['coarse_context'],valid_mask=b.get('valid_mask'),area_weight=b.get('area_weight'));total,parts=build_total_loss(p,b,c['loss'])
  row=dict(batch=i,total=float(total));row.update({k:float(v) for k,v in parts.items() if torch.is_tensor(v) and v.numel()==1});rows.append(row)
pd.DataFrame(rows).to_csv(O/'tables/loss_scale_checkpoint_batches.csv',index=False)
(O/'audit/loss_audit_definition.md').write_text('Forward only; no optimizer, training or checkpoint change. Paper M4/project M5 validation-best checkpoint. 192 evenly spaced indices across frozen training tiles; 48 deterministic batches of 4. Raw objective terms are not directly comparable to mm RMSE. This is an explicit new diagnostic sampling scheme, not a rerun of old randomly initialized audit.\n')
print('LOSS AUDIT COMPLETE',len(rows))

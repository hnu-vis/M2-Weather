"""Bounded Global-shape preflight, run only on an exclusively assigned free GPU."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.Corrformer import Model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    # Leave space for CUDA/cuDNN non-allocator overhead on the dedicated GPU.
    torch.cuda.set_per_process_memory_fraction(0.90)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    configs=SimpleNamespace(node_num=2504,num_nodes=2504,node_list=[8,313],seq_len=48,label_len=24,pred_len=72,enc_in=4,dec_in=4,c_out=4,output_attention=False,moving_avg=25,root_path='./dataset/M3/Global',station_coords_path='./dataset/M3/Global/station_coords.npy',d_model=256,n_heads=8,d_ff=512,e_layers=2,d_layers=1,dec_tcn_layers=1,factor_temporal=1,factor_spatial=1,dropout=.1,activation='relu',embed='timeF',freq='h')
    model=Model(configs).float().cuda().train()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    x=torch.randn(1,48,2504,4,device='cuda')
    dec=torch.randn(1,96,2504,4,device='cuda')
    marks=torch.zeros(1,48,4,device='cuda');decmarks=torch.zeros(1,96,4,device='cuda')
    records=[]
    for step in range(2):
        started=time.time();optimizer.zero_grad(set_to_none=True)
        out=model(x,marks,dec,decmarks)
        loss=out.square().mean();loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();torch.cuda.synchronize()
        record=dict(step=step,loss=loss.item(),seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        records.append(record);print('CORRFORMER_PREFLIGHT '+json.dumps(record),flush=True)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        out=model(x,marks,dec,decmarks)
        assert out.shape==(1,72,2504,4) and torch.isfinite(out).all()
    report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(dict(passed=True,records=records),indent=2))
    print('CORRFORMER_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':main()

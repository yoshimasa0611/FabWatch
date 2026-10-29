"""Render one segment. Splitting into scenes avoids long notebook/runtime timeouts."""
import sys, argparse, json
from pathlib import Path
import pandas as pd
import cv2
from make_video import FPS,W,H,scene_frame,ds_frame,SCENE_W,SCENE_H
from make_video import simulate_route,Fab3D

p=argparse.ArgumentParser();p.add_argument('scene');p.add_argument('start',type=int);p.add_argument('end',type=int);p.add_argument('--root',default=None)
a=p.parse_args()
root=Path(a.root) if a.root else Path(__file__).resolve().parents[1]
out=root/'outputs';d=out/'segments';d.mkdir(exist_ok=True)
met=json.loads((out/'metrics.json').read_text());routes=json.loads((out/'routes.json').read_text())
risk=pd.read_csv(out/'demo_ranking.csv');demo=pd.read_csv(out/'demo_equipment_logs.csv')
traces={k:simulate_route(v) for k,v in routes.items()}
s3d=Fab3D(SCENE_W,SCENE_H) if a.scene in ('intro','fixed','ai') else None
file=d/f'{a.scene}_{a.start:03d}_{a.end:03d}.mp4'
w=cv2.VideoWriter(str(file),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H))
if not w.isOpened():raise RuntimeError('Could not create '+str(file))
for j in range(a.start*FPS,a.end*FPS):
 t=j/FPS
 if s3d:fr=scene_frame(a.scene,t,risk,met,routes,demo,traces,s3d)
 else:fr=ds_frame(a.scene,t,risk,met,out)
 w.write(fr)
w.release()
print('ok',file,flush=True)

"""Reproducibility and submission format checks for the VTK 3D edition."""
from pathlib import Path
import sys,json
import pandas as pd
import cv2, numpy as np
import nbformat
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fabwatch import build_model,eval_scenarios,plan_path,segment_clear,collides
from robot_motion import simulate_route
from scene3d import Fab3D

def test_model():
    model,scaler,thr,train,cal=build_model()
    ev,metrics=eval_scenarios(model,scaler,thr,n_scenarios=8)
    assert len(ev)==64
    m=json.loads((ROOT/'outputs/metrics.json').read_text())
    assert m['n_machine_windows']==480
    assert (m['tp'],m['fp'],m['fn'],m['tn'])==(119,14,0,347)
    assert m['eval_avg_ai_first_fault_s']<m['eval_avg_fixed_first_fault_s']

def test_routes():
    for a,b in [('DOCK','EQ01'),('DOCK','EQ06'),('EQ06','EQ07')]:
        p=plan_path(a,b)
        assert all(segment_clear(x,y) for x,y in zip(p[:-1],p[1:]))
    routes=json.loads((ROOT/'outputs/routes.json').read_text())
    for mode in ('fixed','ai'):
        tr=simulate_route(routes[mode]);assert tr
        assert all(not collides(s['x'],s['y']) for s in tr)
        assert sum(s['mode']=='inspect' for s in tr)>0
        assert tr[-1]['t']/(48 if mode=='fixed' else 56)<2.01

def test_renderer():
    scene=Fab3D(420,240)
    scene.set_state({'x':10.65,'y':2.24,'yaw':-.8,'mode':'inspect','t':0,'vl':0,'vr':0,'current':'EQ06'})
    frame=scene.frame()
    assert frame.shape==(240,420,3)
    assert frame.std()>10,frame.std()  # nonblank actual 3D render
    assert scene.sensor_3d(10.65,2.24,-1.57,.58)>0

def test_notebook():
    nb=nbformat.read(ROOT/'FabWatch_Colab.ipynb',as_version=4)
    for name in ('fabwatch.py','robot_motion.py','scene3d.py','make_video.py','make_report.py','render_segment.py','build_submission.py'):
        prefix='%%writefile src/'+name+'\n'
        cells=[c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith(prefix)]
        assert len(cells)==1,(name,len(cells))
        assert cells[0][len(prefix):].rstrip('\n')==(ROOT/'src'/name).read_text().rstrip('\n'),name

def test_video():
    p=ROOT/'outputs/FabWatch_実3D_提出動画_OP_ED付き.mp4'
    c=cv2.VideoCapture(str(p));assert c.isOpened()
    fps=c.get(cv2.CAP_PROP_FPS);n=c.get(cv2.CAP_PROP_FRAME_COUNT)
    wh=(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    c.release();assert wh==(1280,720)
    assert 60<=n/fps<=180,(n,fps)
    print('FINAL VIDEO',round(n/fps,1),'seconds',wh)

if __name__=='__main__':
    test_model();print('PASS model')
    test_routes();print('PASS routes')
    test_renderer();print('PASS actual VTK 3D renderer')
    test_notebook();print('PASS source/notebook parity')
    test_video();print('PASS movie duration')

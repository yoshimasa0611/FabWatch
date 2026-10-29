"""Anomaly scoring of simulated equipment logs and 2D inspection routes.

Isolation Forest sets the inspection order. A* plans each 2D route.
"""
from __future__ import annotations
import json, sqlite3, heapq, math
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import joblib

SEED=20261026
MACHINES=[
    ('EQ01',(1.9,4.75),(1.9,5.73)),('EQ02',(4.0,4.75),(4.0,5.73)),
    ('EQ03',(6.35,4.75),(6.35,5.73)),('EQ04',(8.55,4.75),(8.55,5.73)),
    ('EQ05',(10.65,4.75),(10.65,5.73)),('EQ06',(10.65,2.24),(10.65,1.24)),
    ('EQ07',(7.25,2.24),(7.25,1.24)),('EQ08',(3.7,2.24),(3.7,1.24))]
MACHINE_IDS=[x[0] for x in MACHINES]
POINTS={'DOCK':(0.9,3.49), **{m:p for m,p,_ in MACHINES}}
BOXES=[(m,(cx-.67,cy-.42,cx+.67,cy+.42)) for m,_,(cx,cy) in MACHINES]
# Columns that robots cannot pass through.
BOXES+= [('COLUMN1',(5.3,3.06,5.8,3.95)),('COLUMN2',(9.32,3.06,9.82,3.95))]
GRID=.18; BOUNDS=(.28,11.85,.35,6.65); ROBOT_RADIUS=.18
FEATURES=['temp_C','vibration_mm_s','power_kW']


def _baseline(machine_i,t,rng):
    # Machine offsets + healthy operational fluctuation (all values synthetic).
    phase=2*np.pi*t/48.0
    return np.column_stack([
        38+1.7*(machine_i%3)+1.25*np.sin(phase)+rng.normal(0,.75,len(t)),
        1.25+.10*(machine_i%4)+.13*np.cos(phase)+rng.normal(0,.12,len(t)),
        6.7+.3*(machine_i%5)+.35*np.sin(phase+.8)+rng.normal(0,.24,len(t))])


def _dataset(seed:int, n:int=32, faulted:tuple=(), tag='eval'):
    rng=np.random.default_rng(seed)
    rows=[]
    for i,m in enumerate(MACHINE_IDS):
        t=np.arange(n)
        vals=_baseline(i,t,rng)
        if m in faulted:
            # Sensor drift sustained over the final 60% of this window.
            drift=np.maximum(0,(t-.4*n)/(.6*n))
            vals[:,0]+=6.0*drift
            vals[:,1]+=1.65*drift
            vals[:,2]+=1.2*drift
        for j in range(n):
            rows.append({'scenario':tag,'machine':m,'tick':j,'temp_C':float(vals[j,0]),
                'vibration_mm_s':float(vals[j,1]),'power_kW':float(vals[j,2]),
                'injected_fault':int(m in faulted)})
    return pd.DataFrame(rows)


def build_model():
    # Entire training AND calibration period is healthy. No faults are used to tune.
    train=_dataset(SEED,160,tag='healthy_train')
    normal_cal=pd.concat([_dataset(SEED+100+i,32,tag=f'cal_{i:02d}') for i in range(40)],ignore_index=True)
    scaler=StandardScaler().fit(train[FEATURES])
    model=IsolationForest(n_estimators=120, max_samples=256, contamination='auto',random_state=2026,n_jobs=1)
    model.fit(scaler.transform(train[FEATURES]))
    cal=pd.concat([window_scores(g,model,scaler) for _,g in normal_cal.groupby('scenario')],ignore_index=True)
    # This sets an operational baseline threshold, NOT an optimal F1-based threshold.
    threshold=float(np.quantile(cal['score'],.95))
    return model,scaler,threshold,train,normal_cal


def window_scores(df,model,scaler):
    d=df.copy()
    d['anomaly_score']=-model.score_samples(scaler.transform(d[FEATURES]))
    # The highest-scoring quartile highlights sustained anomalous readings, not one spike.
    result=d.groupby('machine',sort=False).agg(
       score=('anomaly_score',lambda x:float(np.mean(np.sort(np.asarray(x))[-max(1,len(x)//4):]))),
       injected_fault=('injected_fault','max'),
       mean_temp=('temp_C','mean'), mean_vibration=('vibration_mm_s','mean'),
       mean_power=('power_kW','mean')).reset_index()
    return result


def eval_scenarios(model,scaler,threshold,n_scenarios=60):
    rows=[]
    for s in range(n_scenarios):
        r=np.random.default_rng(4000+s)
        n_fault=int(r.integers(1,4))
        faulted=tuple(r.choice(MACHINE_IDS,n_fault,replace=False).tolist())
        test=_dataset(12000+s,32,faulted,tag=f'eval_{s:02d}')
        scored=window_scores(test,model,scaler)
        for q in scored.itertuples():
            rows.append(dict(scenario=s,machine=q.machine,injected_fault=int(q.injected_fault),
                             anomaly_score=float(q.score),predicted=int(q.score>threshold)))
    result=pd.DataFrame(rows)
    y=result.injected_fault.values; p=result.predicted.values
    precision,recall,f1,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    metrics={'n_scenarios':n_scenarios,'n_machine_windows':len(result),
      'precision':float(precision),'recall':float(recall),'f1':float(f1),
      'roc_auc':float(roc_auc_score(y,result.anomaly_score)),
      'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn)}
    return result,metrics


def collides(x,y,margin=ROBOT_RADIUS+.05):
    if not (BOUNDS[0]+margin<=x<=BOUNDS[1]-margin and BOUNDS[2]+margin<=y<=BOUNDS[3]-margin):return True
    for _,(x1,y1,x2,y2) in BOXES:
        if x1-margin<=x<=x2+margin and y1-margin<=y<=y2+margin:return True
    return False


def nearest_node(p):return (int(round((p[0]-BOUNDS[0])/GRID)),int(round((p[1]-BOUNDS[2])/GRID)))
def node_xy(n):return (BOUNDS[0]+n[0]*GRID,BOUNDS[2]+n[1]*GRID)

@lru_cache(maxsize=128)
def plan_path(start_id:str,end_id:str):
    a=POINTS[start_id];b=POINTS[end_id]
    start=nearest_node(a);end=nearest_node(b)
    if start==end:return (a,b)
    pq=[(0.,start)];cost={start:0.};prev={};visited=set()
    while pq:
        _,node=heapq.heappop(pq)
        if node in visited:continue
        visited.add(node)
        if node==end:break
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            next_node=(node[0]+dx,node[1]+dy)
            xx,yy=node_xy(next_node)
            if collides(xx,yy):continue
            if dx*dy and (collides(*node_xy((node[0]+dx,node[1]))) or collides(*node_xy((node[0],node[1]+dy)))):continue
            nc=cost[node]+math.hypot(dx,dy)*GRID
            if nc<cost.get(next_node,float('inf')):
                cost[next_node]=nc;prev[next_node]=node
                heapq.heappush(pq,(nc+GRID*math.hypot(end[0]-next_node[0],end[1]-next_node[1]),next_node))
    if end not in cost:raise ValueError(f'No route {start_id} to {end_id}')
    path=[end];cur=end
    while cur!=start:
        cur=prev[cur];path.append(cur)
    path.reverse()
    pts=[a]+[node_xy(n) for n in path[1:-1]]+[b]
    # Reduce zigzag while retaining collision checks with dense points.
    simplified=[pts[0]];i=0
    while i<len(pts)-1:
        j=len(pts)-1
        while j>i+1:
            if segment_clear(pts[i],pts[j]):break
            j-=1
        simplified.append(pts[j]);i=j
    return tuple(simplified)


def segment_clear(a,b):
    dis=math.dist(a,b);n=max(1,int(math.ceil(dis/.07)))
    for k in range(n+1):
        t=k/n
        if collides(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])):return False
    return True


def route_length(points):return sum(math.dist(a,b) for a,b in zip(points[:-1],points[1:]))


def build_route(order:list[str],speed=.74,inspection_s=6.0):
    segments=[];total_m=0.;t=0.;here='DOCK';first_fault_t=None
    for machine in order:
        p=plan_path(here,machine);dist=route_length(p)
        segments.append({'from':here,'to':machine,'polyline':p,'start_s':t,'end_s':t+dist/speed,'inspection_end_s':t+dist/speed+inspection_s,'distance_m':dist})
        t+=dist/speed+inspection_s;total_m+=dist;here=machine
    return segments,total_m,t


def first_fault_time(order,segments,faults):
    for machine,seg in zip(order,segments):
        if machine in faults:return float(seg['inspection_end_s'])
    return None


def run_all(out='outputs', n_scenarios=60):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model,scaler,threshold,train,cal=build_model()
    records,metrics=eval_scenarios(model,scaler,threshold,n_scenarios=n_scenarios)
    train.to_csv(out/'healthy_training.csv',index=False)
    records.to_csv(out/'heldout_evaluation.csv',index=False)
    # Demo scenario is excluded from the aggregate evaluation; the anomaly labels are synthetic.
    demo=_dataset(118001,36,('EQ06','EQ07'),tag='demo')
    results=window_scores(demo,model,scaler)
    results['predicted_anomaly']=(results['score']>threshold).astype(int)
    demo.to_csv(out/'demo_equipment_logs.csv',index=False)
    results.to_csv(out/'demo_ranking.csv',index=False)
    with sqlite3.connect(out/'equipment_logs.sqlite') as conn:
        demo.to_sql('sensor_logs',conn,index=False,if_exists='replace')
        results.to_sql('anomaly_ranking',conn,index=False,if_exists='replace')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_machine_time ON sensor_logs(machine,tick)')
    # No target-label access at scheduling time. AI order uses only model scores.
    fixed=MACHINE_IDS[:]
    ai=results.sort_values('score',ascending=False)['machine'].tolist()
    routes={}
    for name,order in [('fixed',fixed),('ai',ai)]:
        segments,d,t=build_route(order)
        routes[name]={'order':order,'segments':segments,'distance_m':round(d,3),
            'duration_s':round(t,3), 'first_fault_inspected_s':round(first_fault_time(order,segments,{'EQ06','EQ07'}),3)}
    # Compare schedules over held-out, independently seeded synthetic scenarios.
    # The true fault labels are used only to score the routes after scheduling.
    simrows=[]
    for sid,group in records.groupby('scenario',sort=True):
        hazards=set(group.loc[group.injected_fault==1,'machine'])
        ai_order=group.sort_values('anomaly_score',ascending=False)['machine'].tolist()
        ss,sd,st=build_route(ai_order)
        fs,fd,ft=build_route(fixed)
        simrows.append({'scenario':int(sid),'fault_count':len(hazards),
           'fixed_first_fault_s':first_fault_time(fixed,fs,hazards),
           'ai_first_fault_s':first_fault_time(ai_order,ss,hazards),
           'fixed_route_m':fd,'ai_route_m':sd,
           'fixed_total_s':ft,'ai_total_s':st})
    simdf=pd.DataFrame(simrows)
    simdf.to_csv(out/'schedule_comparison.csv',index=False)
    diff=simdf.fixed_first_fault_s-simdf.ai_first_fault_s
    metrics['eval_avg_fixed_first_fault_s']=float(simdf.fixed_first_fault_s.mean())
    metrics['eval_avg_ai_first_fault_s']=float(simdf.ai_first_fault_s.mean())
    metrics['eval_mean_first_fault_gain_s']=float(diff.mean())
    metrics['eval_first_fault_improved_n']=int((diff>1e-6).sum())
    metrics['eval_first_fault_worse_n']=int((diff<-1e-6).sum())
    metrics['eval_avg_fixed_route_m']=float(simdf.fixed_route_m.mean())
    metrics['eval_avg_ai_route_m']=float(simdf.ai_route_m.mean())
    rg=np.random.default_rng(5561)
    idx=rg.integers(0,len(diff),size=(4000,len(diff)))
    metrics['paired_bootstrap_gain_ci95_s']=[float(x) for x in np.percentile(diff.to_numpy()[idx].mean(axis=1),[2.5,97.5])]
    metrics['threshold']=threshold
    metrics['demo_faults']=['EQ06','EQ07']
    metrics['demo_fixed_first_fault_s']=routes['fixed']['first_fault_inspected_s']
    metrics['demo_ai_first_fault_s']=routes['ai']['first_fault_inspected_s']
    metrics['demo_fixed_distance_m']=routes['fixed']['distance_m']
    metrics['demo_ai_distance_m']=routes['ai']['distance_m']
    metrics['demo_fixed_duration_s']=routes['fixed']['duration_s']
    metrics['demo_ai_duration_s']=routes['ai']['duration_s']
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'routes.json').write_text(json.dumps(routes,indent=2,ensure_ascii=False),encoding='utf-8')
    joblib.dump({'model':model,'scaler':scaler,'threshold':threshold,'features':FEATURES},out/'isolation_forest.joblib')
    return metrics

if __name__=='__main__':
    m=run_all()
    print(json.dumps(m,indent=2,ensure_ascii=False))

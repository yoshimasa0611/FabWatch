"""Render a true 3D VTK factory scene and Japanese 2D informational overlays.

The model executes 2D differential-drive kinematics and poses 3D meshes.
It is not a rigid-body physics simulation. AI selects equipment order only.
"""
from __future__ import annotations
import json, math, os, subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from fabwatch import MACHINE_IDS,POINTS
from robot_motion import simulate_route,get_state
from scene3d import Fab3D

W,H,FPS=1280,720,10
SCENE_W,SCENE_H=860,490
JP_REG='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
JP_BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
DARK=(15,28,39); PANEL=(24,42,54); CYAN=(83,218,219); ORANGE=(252,192,95)
RED=(241,119,111); GREEN=(107,216,163); WHITE=(235,245,246); MUTED=(161,180,192)

def jp_font(size,bold=False):return ImageFont.truetype(JP_BOLD if bold else JP_REG,size)

def panel(draw,rect,fill=PANEL,outline=(58,84,96),radius=10):
    draw.rounded_rectangle(rect,radius=radius,fill=fill,outline=outline,width=2)

def txt(d,x,y,s,size=20,color=WHITE,bold=False):
    d.text((int(x),int(y)),str(s),font=jp_font(size,bold),fill=color)

def base(title,small):
    im=Image.new('RGB',(W,H),DARK)
    d=ImageDraw.Draw(im)
    d.rectangle((0,0,W,80),fill=(12,25,36))
    txt(d,28,12,'FABWATCH',34,bold=True)
    txt(d,278,20,title,23,CYAN,bold=True)
    txt(d,1040,28,small,15,MUTED)
    d.line((28,77,1250,77),fill=CYAN,width=2)
    return im,d

def main_3d(im,scene_img):
    x,y=25,95
    rgb=cv2.cvtColor(scene_img,cv2.COLOR_BGR2RGB)
    im.paste(Image.fromarray(rgb).resize((SCENE_W,SCENE_H)),(x,y))
    d=ImageDraw.Draw(im);d.rectangle((x,y,x+SCENE_W,y+SCENE_H),outline=(99,143,155),width=2)
    return d

def status_box(d,state,rank,mode=''):
    panel(d,(899,96,1252,585))
    txt(d,923,108,'設備の異常スコア',22,bold=True)
    rows=rank.sort_values('score',ascending=False)
    for i,r in enumerate(rows.itertuples()):
        yy=151+i*37
        current=r.machine==state.get('current')
        if current:d.rounded_rectangle((910,yy-5,1242,yy+31),fill=(48,73,82),radius=4)
        txt(d,922,yy,r.machine,19,WHITE,current)
        xx=994; bw=155
        d.rectangle((xx,yy+11,xx+bw,yy+23),fill=(47,65,72))
        v=max(.02,min(1,(float(r.score)-.42)/.29))
        cc=RED if r.predicted_anomaly else CYAN
        d.rectangle((xx,yy+11,xx+int(bw*v),yy+23),fill=cc)
        txt(d,1164,yy,f'{r.score:.3f}',16,cc)
    d.line((913,466,1238,466),fill=(74,96,107),width=1)
    txt(d,916,475,'ロボットの動作',19,WHITE,True)
    action='点検停止中' if state['mode']=='inspect' else '移動中'
    txt(d,916,505,f'状態：{action}',16)
    txt(d,916,534,f'左右モータ：{state["vl"]:.1f} / {state["vr"]:.1f} rad/s',14,MUTED)
    txt(d,916,559,f'距離（前方）：{state["s_front"]:.2f} m',15,ORANGE)

def footer(im,main,sub):
    d=ImageDraw.Draw(im)
    panel(d,(25,604,1252,696),fill=(21,37,49),radius=7)
    txt(d,46,614,main,26,WHITE,True)
    txt(d,46,652,sub,18,MUTED)
    return im

def scene_camera(s3d,state,mode):
    cam=s3d.ren.GetActiveCamera()
    if mode in ('fixed','ai'):
        x,y=state['x'],state['y']
        positive=state.get('current') in ('EQ06','EQ07','EQ08')
        yy=y+(4.5 if positive else -4.5)
        cam.SetPosition(x+2.9,yy,3.6)
        cam.SetFocalPoint(x,y,.32)
        cam.SetViewUp(0,0,1)
        cam.SetViewAngle(44)
    else:
        cam.SetPosition(13.3,-10.0,10.2)
        cam.SetFocalPoint(6.05,3.4,.35)
        cam.SetViewUp(0,0,1)
        cam.SetViewAngle(35)
    s3d.ren.ResetCameraClippingRange()

def highlight_points(im,state):
    if state['mode']!='inspect':return
    d=ImageDraw.Draw(im)
    d.rounded_rectangle((45,504,410,575),fill=(20,69,63),outline=CYAN,width=2,radius=7)
    txt(d,62,515,f'点検中：{state["current"]}',28,WHITE,True)
    p=float(state.get('inspect_progress',0))
    d.rectangle((62,555,376,567),fill=(54,89,91))
    d.rectangle((62,555,62+int(314*p),567),fill=CYAN)

def scene_frame(section,elapsed,risk,met,routes,demo,traces,s3d):
    if section in ('fixed','ai'):
        screen_seconds=48 if section=='fixed' else 56
        state,simt,total_t=get_state(traces[section],elapsed,screen_seconds)
        heading='固定順巡回' if section=='fixed' else 'AI異常度優先巡回'
        playback=total_t/screen_seconds
    else:
        state=traces['ai'][0]
        heading='仮想設備点検エリア'
        simt=0;playback=1
    s3d.set_state(state,elapsed)
    scene_camera(s3d,state,section)
    image,d=base(heading,'3Dレンダリング：VTK')
    d=main_3d(image,s3d.frame())
    status_box(d,state,risk,section)
    if section in ('fixed','ai'):
        highlight_points(image,state)
        first=routes[section]['first_fault_inspected_s']
        dist=routes[section]['distance_m']
        primary='固定方式：設備番号順に巡回' if section=='fixed' else 'AI方式：異常スコアの高い設備から巡回'
        secondary=f'再生約{playback:.2f}倍   |   初回異常点検 {first:.1f}秒   |   計画移動距離 {dist:.1f}m'
        footer(image,primary,secondary)
    else:
        footer(image,'異常検知結果から点検先を決定','8設備の模擬ログを分析し、AIのスコア順で点検ルートを変更')
    return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

def ds_frame(section,elapsed,rank,met,out):
    headings={'train':'学習・閾値設定','validate':'未学習データでの評価','results':'点検順の比較結果','limitations':'適用範囲と改善点'}
    image,d=base(headings[section],'FABWATCH / DATA ANALYSIS')
    panel(d,(24,97,1253,584))
    if section in ('train','validate'):
        pic='sensor_signals.png' if section=='train' else 'score_distribution.png'
        chart=Image.open(out/pic).convert('RGB')
        chart.thumbnail((835,420))
        image.paste(chart,(44,125))
        d=ImageDraw.Draw(image)
        panel(d,(905,115,1235,552),fill=(30,51,61))
        if section=='train':
            lines=[('01  正常ログを生成', '温度・振動・電力を使用'),('02  学習と閾値を分離','正常履歴でIsolation Forestを学習'),('03  別の正常40条件','スコアの95%点を閾値とする')]
            foot=('学習データと閾値設定用データを分離','異常ラベルを使って閾値を調整していない。')
        else:
            lines=[('評価条件：60セット','合計480設備窓、1～3台に異常注入'),('適合率 / Precision',f'{met["precision"]:.3f}'),('再現率 / Recall',f'{met["recall"]:.3f}'),('F1スコア',f'{met["f1"]:.3f}')]
            foot=('学習していない60条件で異常検知を評価','評価結果は人工データのドリフト検出性能であり実機の故障予測ではない。')
        for i,(a,b) in enumerate(lines):
            yy=139+i*(90 if section=='train' else 80)
            txt(d,919,yy,a,20,CYAN,True)
            txt(d,919,yy+32,b,16,WHITE)
        footer(image,*foot)
    elif section=='results':
        txt(d,55,117,'初回異常設備の点検完了までの平均時間',26,WHITE,True)
        barwidth=650
        for i,(label,sec,color) in enumerate([('固定方式',met['eval_avg_fixed_first_fault_s'],MUTED),('AI方式',met['eval_avg_ai_first_fault_s'],CYAN)]):
            y=205+i*108
            txt(d,62,y,label,24,WHITE,True)
            d.rectangle((240,y,240+barwidth,y+37),fill=(39,61,71))
            d.rectangle((240,y,240+int(barwidth*sec/35),y+37),fill=color)
            txt(d,940,y-3,f'{sec:.1f}秒',26,color,True)
        txt(d,75,449,f'AIが早かった条件：{met["eval_first_fault_improved_n"]} / 60',23,CYAN,True)
        txt(d,75,499,f'平均移動距離：固定 {met["eval_avg_fixed_route_m"]:.1f}m  /  AI {met["eval_avg_ai_route_m"]:.1f}m',22,ORANGE)
        footer(image,'初回点検は早まったが、移動距離は増えた','移動時間を考慮しないスコア順巡回には改善の余地がある。')
    else:
        content=[
          '・設備データ、工場配置、異常注入条件はいずれも人工的に設定した。',
          '・AIが決めるのは点検順序。モーター制御は差動二輪の簡略運動モデル。',
          '・映像はVTKの立体形状と透視投影を使う実際の3Dレンダリング。',
          '・剛体力学エンジンによる摩擦・衝突・荷重などの実機再現ではない。',
          '・今後は移動時間を含めた最適化と、実際の計測ログによる検証が必要。'
        ]
        for i,s in enumerate(content):txt(d,64,155+i*76,s,23,WHITE)
        footer(image,'FabWatch / 最終課題','異常検知・点検順序・ロボットの動作を分けて設計した。')
    return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

def make_main_video(root:Path,fps=FPS):
    out=root/'outputs';out.mkdir(parents=True,exist_ok=True)
    risk=pd.read_csv(out/'demo_ranking.csv');demo=pd.read_csv(out/'demo_equipment_logs.csv')
    metrics=json.loads((out/'metrics.json').read_text())
    routes=json.loads((out/'routes.json').read_text())
    traces={k:simulate_route(v) for k,v in routes.items()}
    s3d=Fab3D(SCENE_W,SCENE_H)
    # 6 FPS intentionally balances rendering cost and Colab CPU accessibility.
    scenes=[('intro',6),('train',13),('validate',11),('fixed',48),('ai',56),('results',13),('limitations',6)]
    main=out/'FabWatch_実3D_本編.mp4'
    writer=cv2.VideoWriter(str(main),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
    if not writer.isOpened():raise RuntimeError('Cannot create MP4')
    for section,seconds in scenes:
        for n in range(int(seconds*fps)):
            if section in ('train','validate','results','limitations'):
                frame=ds_frame(section,n/fps,risk,metrics,out)
            else:
                frame=scene_frame(section,n/fps,risk,metrics,routes,demo,traces,s3d)
            writer.write(frame)
        print('rendered',section,flush=True)
    writer.release()
    return main

def add_op_ed(root:Path,main:Path,op:Path,ed:Path):
    """Use ffmpeg concat with fps normalization; preserves 3-second OP/ED durations."""
    output=root/'outputs'/'FabWatch_実3D_提出動画_OP_ED付き.mp4'
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(op),'-i',str(main),'-i',str(ed),
         '-filter_complex',f'[0:v]fps={FPS},scale={W}:{H},setsar=1[v0];[1:v]fps={FPS},scale={W}:{H},setsar=1[v1];[2:v]fps={FPS},scale={W}:{H},setsar=1[v2];[v0][v1][v2]concat=n=3:v=1:a=0[v]',
         '-map','[v]','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    subprocess.run(cmd,check=True)
    return output

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    main=make_main_video(root)
    op,ed=root/'assets'/'OP.mp4',root/'assets'/'ED.mp4'
    if op.exists() and ed.exists():print('submission',add_op_ed(root,main,op,ed))
    else:print('main only',main)

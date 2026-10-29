"""Rebuild metrics, charts, Japanese PDF and 3D movie. Supports resuming completed video segments."""
import sys, subprocess, os, cv2, math
from pathlib import Path
from make_video import add_op_ed, FPS
from make_report import build_pdf, make_charts
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
segments=[('intro',0,6),('train',0,13),('validate',0,11),
          ('fixed',0,16),('fixed',16,32),('fixed',32,48),
          ('ai',0,14),('ai',14,28),('ai',28,42),('ai',42,56),
          ('results',0,13),('limitations',0,6)]

def run(cmd):
    print('RUN', ' '.join(str(x) for x in cmd), flush=True)
    subprocess.run([str(a) for a in cmd],check=True,cwd=ROOT)

def valid_segment(path, seconds):
    """An interrupted MP4 may exist but have no playable frames."""
    if not path.exists() or path.stat().st_size < 10000:
        return False
    cap=cv2.VideoCapture(str(path))
    ok=cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS) if ok else 0
    frames=cap.get(cv2.CAP_PROP_FRAME_COUNT) if ok else 0
    cap.release()
    return bool(ok and fps>0 and frames>=seconds*fps-2)

if __name__=='__main__':
    # Recalculate model and numerical metrics from the same fixed-seed source.
    run([sys.executable,ROOT/'src/fabwatch.py'])
    # Important: charts MUST be created before train/validate video segments.
    make_charts(OUT)
    pdf=build_pdf(OUT,account_id=os.environ.get('OMNI_ACCOUNT_ID','ACCOUNT_NAME'))
    print('PDF prepared:',pdf, flush=True)
    for scene,a,b in segments:
        path=OUT/'segments'/f'{scene}_{a:03d}_{b:03d}.mp4'
        if valid_segment(path, b-a):
            print('SKIP valid segment:',path.name,flush=True)
            continue
        if path.exists():path.unlink()
        run([sys.executable,ROOT/'src/render_segment.py',scene,a,b])
        if not valid_segment(path,b-a):
            raise RuntimeError('Rendering did not produce a valid segment: '+str(path))
    ff_list=OUT/'segments'/'concat_list.txt'
    ff_list.write_text(''.join("file '"+str(OUT/'segments'/f'{sc}_{a:03d}_{b:03d}.mp4')+"'\n" for sc,a,b in segments),encoding='utf-8')
    movie=OUT/'FabWatch_実3D_本編.mp4'
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',ff_list,'-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p',movie])
    cap=cv2.VideoCapture(str(movie));cap.set(cv2.CAP_PROP_POS_MSEC,94*1000)
    ok,frame=cap.read();cap.release()
    if ok:cv2.imwrite(str(OUT/'3d_inspection.png'),frame)
    # Rebuild report after screenshot extraction to include the true 3D preview.
    pdf=build_pdf(OUT,account_id=os.environ.get('OMNI_ACCOUNT_ID','ACCOUNT_NAME'))
    print('Final PDF with 3D preview:',pdf,flush=True)
    op,ed=ROOT/'assets/OP.mp4',ROOT/'assets/ED.mp4'
    if op.exists() and ed.exists():
        print('FINAL',add_op_ed(ROOT,movie,op,ed),flush=True)
    else:
        print('OP/ED not present. Main:',movie,flush=True)

"""Differential-drive kinematic tracking and 2D range ray calculations.
Used only for pose generation; 3D visualization is rendered separately via VTK.
"""
import math
from fabwatch import BOXES,POINTS,BOUNDS
WHEEL_R=0.06
AXLE=0.28
ROBOT_L=.42
ROBOT_W=.28
ROBOT_H=.14
MAX_SENSOR=1.35

def clamp(x,a,b):return max(a,min(b,x))

def ray_box_intersection(origin, theta, rect, max_range=MAX_SENSOR):
    x0,y0=origin
    x1,y1,x2,y2=rect
    step=0.03; s=0.0
    while s<=max_range:
        x=x0+math.cos(theta)*s; y=y0+math.sin(theta)*s
        if x1<=x<=x2 and y1<=y<=y2:
            return s
        s+=step
    s=0.0
    while s<=max_range:
        x=x0+math.cos(theta)*s; y=y0+math.sin(theta)*s
        if x<BOUNDS[0] or x>BOUNDS[1] or y<BOUNDS[2] or y>BOUNDS[3]:
            return s
        s+=step
    return max_range


def sensor_distances(x,y,yaw):
    sensors={'s_front':MAX_SENSOR,'s_left':MAX_SENSOR,'s_right':MAX_SENSOR}
    rays={'s_front':yaw,'s_left':yaw+math.radians(35),'s_right':yaw-math.radians(35)}
    for key,ang in rays.items():
        best=MAX_SENSOR
        for _,rect in BOXES:
            best=min(best, ray_box_intersection((x,y),ang,rect,MAX_SENSOR))
        sensors[key]=best
    return sensors


def simulate_route(route,speed=0.74,inspection_s=6.0,dt=0.1):
    x,y=POINTS['DOCK']; yaw=0.0; t=0.0; visited=[]; states=[]
    v_nom=speed
    for seg in route['segments']:
        polyline=[tuple(p) for p in seg['polyline']]
        target_machine=seg['to']
        waypoint_idx=1
        while waypoint_idx < len(polyline):
            tx,ty=polyline[waypoint_idx]
            if math.dist((x,y),(tx,ty))<0.12 and waypoint_idx < len(polyline)-1:
                waypoint_idx += 1; tx,ty=polyline[waypoint_idx]
            dx=tx-x; dy=ty-y
            desired=math.atan2(dy,dx)
            err=(desired-yaw+math.pi)%(2*math.pi)-math.pi
            v=v_nom*clamp(1-0.35*abs(err),0.22,1.0)
            if math.dist((x,y),(tx,ty))<0.18 and waypoint_idx==len(polyline)-1:
                v=min(v,0.28)
            omega=clamp(2.6*err,-1.8,1.8)
            vl=v-omega*AXLE/2; vr=v+omega*AXLE/2
            v_robot=(vr+vl)/2; w_robot=(vr-vl)/AXLE
            x += v_robot*math.cos(yaw)*dt; y += v_robot*math.sin(yaw)*dt; yaw += w_robot*dt
            sens=sensor_distances(x,y,yaw)
            states.append({'t':t,'x':x,'y':y,'yaw':yaw,'current':target_machine,'visited':visited.copy(),
                           'mode':'move','vl':vl/WHEEL_R,'vr':vr/WHEEL_R,**sens})
            t+=dt
            if waypoint_idx==len(polyline)-1 and math.dist((x,y),polyline[-1])<0.08:
                break
        x,y=polyline[-1]
        equip=[r for r in BOXES if r[0]==target_machine][0][1]
        cx=(equip[0]+equip[2])/2; cy=(equip[1]+equip[3])/2; yaw=math.atan2(cy-y,cx-x)
        steps=max(1,int(round(inspection_s/dt)))
        for k in range(steps):
            sens=sensor_distances(x,y,yaw)
            states.append({'t':t,'x':x,'y':y,'yaw':yaw,'current':target_machine,'visited':visited.copy(),
                           'mode':'inspect','inspect_progress':(k+1)/steps,'vl':0.0,'vr':0.0,**sens})
            t+=dt
        visited=visited+[target_machine]
    for _ in range(int(2/dt)):
        sens=sensor_distances(x,y,yaw)
        states.append({'t':t,'x':x,'y':y,'yaw':yaw,'current':None,'visited':visited.copy(),'mode':'done','vl':0.0,'vr':0.0,**sens})
        t+=dt
    return states


def get_state(trace, elapsed, playback_duration):
    total_t=trace[-1]['t'] if trace else 1.0
    sim_t=clamp(elapsed/playback_duration,0.0,1.0)*total_t
    idx=min(len(trace)-1, int(sim_t/0.1))
    return trace[idx], sim_t, total_t



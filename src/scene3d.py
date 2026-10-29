"""Actual 3D geometry and perspective camera rendered with VTK.

3D visual simulation: wheel rotation and chassis pose follow differential-drive
kinematics, not an articulated rigid-body physics solver. Distance readings are
3D horizontal rays intersecting equipment bounding boxes.
"""
from __future__ import annotations
import math
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from fabwatch import BOXES, BOUNDS


def rgb(r,g,b): return tuple(float(v)/255 for v in (r,g,b))

class Fab3D:
    def __init__(self,width=800,height=450):
        self.width,self.height=width,height
        self.ren=vtk.vtkRenderer()
        self.ren.SetBackground(0.064,0.096,0.13)
        self.ren.GradientBackgroundOn()
        self.ren.SetBackground2(0.22,0.29,0.32)
        self.win=vtk.vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.SetSize(width,height)
        self.win.AddRenderer(self.ren)
        self.ren.SetUseFXAA(False)
        self._build_plant()
        self._build_robot()
        self._set_camera()
        self.window_filter=vtk.vtkWindowToImageFilter()
        self.window_filter.SetInput(self.win)
        self.window_filter.SetInputBufferTypeToRGB()
        self.ren.ResetCameraClippingRange()
        self.win.Render()

    def mat(self,actor,c,ambient=0.25,diffuse=0.72,specular=0.18):
        p=actor.GetProperty();p.SetColor(*rgb(*c));p.SetAmbient(ambient);p.SetDiffuse(diffuse)
        p.SetSpecular(specular);p.SetSpecularPower(23)
        return actor
    def box(self,pos,dims,c):
        s=vtk.vtkCubeSource();s.SetXLength(dims[0]);s.SetYLength(dims[1]);s.SetZLength(dims[2])
        m=vtk.vtkPolyDataMapper();m.SetInputConnection(s.GetOutputPort())
        a=vtk.vtkActor();a.SetMapper(m);a.SetPosition(*pos);self.mat(a,c);self.ren.AddActor(a)
        return a
    def cylinder(self,pos,radius,height,c):
        s=vtk.vtkCylinderSource();s.SetRadius(radius);s.SetHeight(height);s.SetResolution(18)
        m=vtk.vtkPolyDataMapper();m.SetInputConnection(s.GetOutputPort())
        a=vtk.vtkActor();a.SetMapper(m);a.SetPosition(*pos);self.mat(a,c);self.ren.AddActor(a)
        return a
    def sphere(self,pos,r,c):
        s=vtk.vtkSphereSource();s.SetRadius(r);s.SetThetaResolution(16);s.SetPhiResolution(12)
        m=vtk.vtkPolyDataMapper();m.SetInputConnection(s.GetOutputPort())
        a=vtk.vtkActor();a.SetMapper(m);a.SetPosition(*pos);self.mat(a,c);self.ren.AddActor(a);return a
    def label(self,text,pos,scale=0.14,c=(245,247,247)):
        s=vtk.vtkVectorText();s.SetText(text)
        m=vtk.vtkPolyDataMapper();m.SetInputConnection(s.GetOutputPort())
        a=vtk.vtkFollower();a.SetMapper(m);a.SetScale(scale,scale,scale);a.SetPosition(*pos)
        self.mat(a,c,ambient=0.7,diffuse=0.3,specular=0)
        self.ren.AddActor(a);a.SetCamera(self.ren.GetActiveCamera());return a
    def line_actor(self,points,c,width=2):
        poly=vtk.vtkPolyData();pts=vtk.vtkPoints()
        for v in points:pts.InsertNextPoint(*v)
        poly.SetPoints(pts);lines=vtk.vtkCellArray();lines.InsertNextCell(len(points))
        for i in range(len(points)):lines.InsertCellPoint(i)
        poly.SetLines(lines)
        m=vtk.vtkPolyDataMapper();m.SetInputData(poly)
        a=vtk.vtkActor();a.SetMapper(m);a.GetProperty().SetColor(*rgb(*c));a.GetProperty().SetLineWidth(width);a.GetProperty().SetAmbient(1)
        self.ren.AddActor(a);return a
    def _build_plant(self):
        # 3D cleanroom production floor, marked inspection aisle, structural elements.
        self.box((6.05,3.5,-.13),(12.2,7.1,.25),(176,189,190))
        self.box((6.05,3.47,.008),(11.9,.82,.015),(87,117,124))
        for y in (2.99,3.92):
            self.box((6.05,y,.024),(11.5,.045,.02),(246,187,62))
        for x in np.arange(.7,11.7,1.05):
            self.box((x,3.48,.022),(.48,.018,.018),(218,230,227))
        for ix in (1,2,3,4,5,6,7,8,9,10,11):
            self.line_actor([(ix,0.25,.02),(ix,6.75,.02)],(143,160,164),1)
        for yy in (1,2,3,4,5,6):
            self.line_actor([(.25,yy,.021),(11.85,yy,.021)],(143,160,164),1)
        # perimeter structural framing, ventilation above equipment
        self.box((6.05,6.78,1.1),(12.2,.10,2.2),(171,183,188))
        for x in (0.35,11.77):
            self.box((x,.25,1.35),(.10,.18,2.7),(114,129,133))

        # equipment cabinets using original collision rectangles exactly
        self.lamps={}
        self.cabinets={}
        for eq,(x1,y1,x2,y2) in BOXES:
            x,y=(x1+x2)/2,(y1+y2)/2
            if eq.startswith('COLUMN'):
                self.box((x,y,.88),(x2-x1,y2-y1,1.76),(115,132,137));continue
            w,d=x2-x1,y2-y1
            c=(205,218,220) if eq not in ('EQ06','EQ07') else (196,210,217)
            self.cabinets[eq]=self.box((x,y,.68),(w,d,1.36),c)
            self.box((x,y2+.012,.73),(w*.72,.025,.76),(64,86,99))
            for z in (.46,.59,.72,.85):self.box((x-.36,y2+.035,z),(.36,.012,.018),(115,159,169))
            self.box((x+.36,y2+.047,.93),(.16,.03,.18),(36,64,71))
            lamp=self.sphere((x2-.15,y1+.15,1.43),.085,(46,193,120))
            self.lamps[eq]=lamp
            self.label(eq,(x-.31,y2+.045,1.21),.18)
            # service cabinet front panel handle
            self.box((x+.42,y2+.048,.7),(.025,.035,.18),(210,221,231))
        # service dock and floor markings
        self.box((.86,3.5,.075),(.86,.65,.15),(70,128,141))
        self.label('DOCK',(.52,3.08,.14),.11)
        for x in (0.5,1.2):
            self.box((x,3.14,.036),(.12,.055,.03),(239,184,50))
        # hazard striped stops, static props not changing navigable footprint
        for x in (5.55,9.57):
            self.box((x,3.16,.04),(.36,.08,.04),(220,164,65))
        light=vtk.vtkLight();light.SetPosition(3,-3,11);light.SetFocalPoint(6,3,0);light.SetIntensity(0.8);self.ren.AddLight(light)
        light2=vtk.vtkLight();light2.SetPosition(14,8,10);light2.SetFocalPoint(4,3,0);light2.SetIntensity(.48);self.ren.AddLight(light2)
    def _build_robot(self):
        self.assembly=vtk.vtkAssembly();self.ren.AddActor(self.assembly)
        def child_box(pos,dims,c):
            s=vtk.vtkCubeSource();s.SetXLength(dims[0]);s.SetYLength(dims[1]);s.SetZLength(dims[2]);m=vtk.vtkPolyDataMapper();m.SetInputConnection(s.GetOutputPort());a=vtk.vtkActor();a.SetMapper(m);a.SetPosition(*pos);self.mat(a,c);self.assembly.AddPart(a);return a
        def child_cyl(pos,r,h,c):
            s=vtk.vtkCylinderSource();s.SetRadius(r);s.SetHeight(h);s.SetResolution(24);m=vtk.vtkPolyDataMapper();m.SetInputConnection(s.GetOutputPort());a=vtk.vtkActor();a.SetMapper(m);a.SetPosition(*pos);self.mat(a,c);self.assembly.AddPart(a);return a
        child_box((0,0,.24),(.50,.32,.22),(35,118,149))
        child_box((.10,0,.38),(.23,.23,.06),(51,76,87))
        child_box((.21,0,.24),(.07,.23,.055),(248,197,73))
        self.wheels=[]
        for side in (-1,1):
            a=child_cyl((0,side*.195,.12),.115,.085,(30,41,47))
            self.wheels.append(a)
        child_cyl((-.19,0,.075),.065,.12,(90,98,103))
        child_box((.055,0,.49),(.09,.09,.12),(97,139,150))
        child_box((.055,0,.57),(.20,.15,.035),(32,66,83))
        self.beams=[]
        for c in ((65,229,220),(252,188,72),(114,224,138)):
            l=vtk.vtkLineSource();m=vtk.vtkPolyDataMapper();m.SetInputConnection(l.GetOutputPort());a=vtk.vtkActor();a.SetMapper(m);a.GetProperty().SetColor(*rgb(*c));a.GetProperty().SetLineWidth(3);a.GetProperty().SetAmbient(1);self.ren.AddActor(a);self.beams.append((l,a))
        self.ring=self.cylinder((0,0,.015),.31,.014,(30,169,195))
        self.ring.VisibilityOff()
        self._wheel_spin=[0.,0.];self._last_sim_t=None
    def _set_camera(self):
        cam=self.ren.GetActiveCamera()
        cam.SetPosition(13.3,-10.0,10.2)
        cam.SetFocalPoint(6.05,3.4,.35)
        cam.SetViewUp(0,0,1)
        cam.SetViewAngle(35)
        self.ren.ResetCameraClippingRange()
    def set_state(self,s,pulse=0):
        x,y,yaw=s['x'],s['y'],s['yaw']
        self.assembly.SetPosition(x,y,0)
        self.assembly.SetOrientation(0,0,math.degrees(yaw))
        t=s.get('t',0.)
        if self._last_sim_t is None:dt=0
        else:dt=max(0,min(.6,t-self._last_sim_t))
        self._last_sim_t=t
        for i,vel in enumerate((s['vl'],s['vr'])):
            self._wheel_spin[i]=(self._wheel_spin[i]+math.degrees(vel*dt))%360
            self.wheels[i].SetOrientation(0,self._wheel_spin[i],0)
        # sensor beams in world XYZ, intersect cabinet volumes at sensor plane z=.58
        z=.58
        for ((src,actor),off) in zip(self.beams,(0,math.radians(36),-math.radians(36))):
            angle=yaw+off;dist=self.sensor_3d(x,y,angle,z)
            src.SetPoint1(x+.06*math.cos(yaw),y+.06*math.sin(yaw),z)
            src.SetPoint2(x+dist*math.cos(angle),y+dist*math.sin(angle),z)
            src.Modified()
        self.ring.SetPosition(x,y,.035)
        self.ring.SetScale(1+.18*math.sin(pulse),1+.18*math.sin(pulse),1)
        if s['mode']=='inspect':self.ring.VisibilityOn()
        else:self.ring.VisibilityOff()
        for eq,lamp in self.lamps.items():
            if eq==s.get('current') and s['mode']=='inspect':
                lamp.GetProperty().SetColor(*rgb(243,198,66))
            else:
                lamp.GetProperty().SetColor(*rgb(*((234,79,82) if eq in ('EQ06','EQ07') else (46,193,120))))
    @staticmethod
    def sensor_3d(x,y,angle,z,max_range=1.50):
        # 3D ray originates above floor; cabinets and columns have known 3D extents.
        origin=np.array([x,y,z]);dx,dy=math.cos(angle),math.sin(angle)
        smallest=max_range
        for _,(x1,y1,x2,y2) in BOXES:
            low=0.; high=max_range
            for o,d,a,b in ((x,dx,x1,x2),(y,dy,y1,y2)):
                if abs(d)<1e-8:
                    if o<a or o>b:high=-1;break
                else:
                    t1=(a-o)/d;t2=(b-o)/d
                    low=max(low,min(t1,t2));high=min(high,max(t1,t2))
            if high>=max(low,0):smallest=min(smallest,max(low,0))
        return smallest
    def frame(self):
        self.win.Render();self.window_filter.Modified();self.window_filter.Update()
        image=self.window_filter.GetOutput();w,h,_=image.GetDimensions()
        data=vtk_to_numpy(image.GetPointData().GetScalars()).reshape(h,w,3)
        return np.ascontiguousarray(data[::-1,:,::-1])

'''************************************************************************
    LightPyCL geometry object and optical elements class file
    Copyright 2014, by Govinda Lilley

    This file is part of LightPyCL.

    LightPyCL is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LightPyCL is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with LightPyCL.  If not, see <http://www.gnu.org/licenses/>.
**************************************************************************'''

import numpy as np
from dxfwrite import DXFEngine as dxf

class GeoObject():
	""" GeoObject provides structure to store meshes and perform simple geomentric operations such as translate and rotate.
	This cass also privides conversion methods required by the tracer kernel and dxf output."""
	verticies=None
	triangles=None
	IOR = 1.0
	reflectivity = 1.0
	dissipation  = 0.0 	#in 1/m
	AR_IOR       = 1.0 	# not in use yet
	AR_thickness = 0.0 	# not in use yet
	anisotropy   = None 	# not in use yet
	
	matTypes = {"refractive":0, "mirror":1, "terminator":2, "measure":3, "refractive_anisotropic":4}
	matType = "refractive"
	

	def __init__(self,verts,tris, mat_type="refractive", IOR=1.0, reflectivity = 1.0, dissipation = 0.0, AR_IOR=1.0, AR_thickness=0.0, anisotropy = None):
		self.verticies = verts
		self.triangles = tris
		self.setMaterial(mat_type,IOR, reflectivity, dissipation, AR_IOR, AR_thickness, anisotropy)
	
	def setMaterial(self,mat_type="refractive", IOR=1.0, reflectivity = 1.0, dissipation = 0.0, AR_IOR=1.0, AR_thickness=0.0, anisotropy = None):
		if mat_type in self.matTypes:
			self.matType = mat_type
		else:
			default = "refractive"
			self.matType = default
			print "Warning: material", mat, "unknown. Setting material as ", default

		if self.matType == "refractive":
			self.IOR = IOR
			self.dissipation = dissipation
			self.AR_IOR = AR_IOR
			self.AR_thickness = AR_thickness
		elif self.matType == "refractive":
			self.reflectivity = reflectivity
		elif self.matType == "refractive_anisotropic":
			print "Warning: anisotropic materials are not yet supported."
		
	def getMaterialBuf(self):
		""" returns a list of material parameters for the tracer code"""
		return {"type":self.matTypes.get(self.matType), "IOR":self.IOR, "R":self.reflectivity, "dissipation":self.dissipation}
	
	def translate(self,vec):
		self.verticies = self.verticies + np.array(vec)
	
	
	def rotate(self,axis="x",angle=np.pi/2,pivot = (0,0,0,0)):
		Rx = lambda x: np.matrix([[1,0,0,0],[0,np.cos(x),-np.sin(x),0],[0,np.sin(x),np.cos(x),0],[0,0,0,0]])
		Ry = lambda x: np.matrix([[np.cos(x),0,np.sin(x),0],[0,1,0,0],[-np.sin(x),0,np.cos(x),0],[0,0,0,0]])
		Rz = lambda x: np.matrix([[np.cos(x),-np.sin(x),0,0],[np.sin(x),np.cos(x),0,0],[0,0,1,0],[0,0,0,0]])
		if axis in {"x","X"}:
			R=Rx
		elif axis in {"y","Y"}:
			R=Ry
		elif axis in {"z","Z"}:
			R=Rz
		else:
			R=Rx
		
		#print self.verticies.dtype
		vtmp = list(np.transpose(R(angle) * np.transpose(np.array(self.verticies) - np.array(pivot))) + np.array(pivot))
		
		i=0
		for vert_ in vtmp:
			vert = np.array(vert_)
			self.verticies[i] = np.array([vert[0][0],vert[0][1],vert[0][2],0],dtype=np.float32)
			#print self.verticies[i]
			i+=1

	def trimesh(self):
		""" returns a list tmesh of triangles that themselves are composed of a list of three verticies."""
		tmesh = []
		for tri in self.triangles:
			tri_verts = []
			for idx in tri:
				tri_verts.append(self.verticies[idx])
			tmesh.append(tri_verts)
		return tmesh
	
	def tribuf(self):
		""" 	generate 3 buffers for PyOpenCL tracer code
			each buffer contains one vertex of a triangle.
			therefore m_v0[i], m_v1[i] and m_v2[i] constitutes a triangle"""
		m_v0 = []
		m_v1 = []
		m_v2 = []
		for tri in self.triangles:
			m_v0.append(self.verticies[tri[0]])
			m_v1.append(self.verticies[tri[1]])
			m_v2.append(self.verticies[tri[2]])
		return (m_v0,m_v1,m_v2)	
	
	def write_dxf(self,dxf_file):
		""" writes geometry to a file by using the dxfwrite libs."""
		drawing = dxf.drawing(dxf_file)
		drawing.add_layer('0', color=2)
		for tri in self.triangles:
			drawing.add(dxf.face3d([self.verticies[tri[0]][0:3], 
						self.verticies[tri[1]][0:3], 
						self.verticies[tri[2]][0:3]], layer="0"))
		#drawing.add(dxf.text('Test', insert=(0, 0.2), layer='TEXTLAYER'))
		drawing.save()	

		
class optical_elements():
	""" The optical_elements class provides generator methods that create mesh objects from basic parametrs of the corresponding optical elements.
	optical element generators return a GeoObject with verticies triangle indicies and index of refraction properties."""
	def cube(self, center, size):
		verts = np.array(	[[-1,-1,-1,0],
					 [ 1,-1,-1,0],
					 [-1, 1,-1,0],
					 [ 1, 1,-1,0],
					 [-1,-1, 1,0],
					 [ 1,-1, 1,0],
					 [-1, 1, 1,0],
					 [ 1, 1, 1,0]],
					 dtype=np.float32)/2.0*size + center
		triangles = [[0,1,2],[2,3,1], #bottom
			     [4,5,6],[6,7,5], #top
			     [0,1,4],[4,5,1], #rear
			     [2,3,6],[6,7,3], #front
			     [0,2,4],[4,6,2], #left
			     [1,3,5],[5,7,3]] #right
		mesh = GeoObject(verts,triangles)
		return mesh

	def spherical_lens_nofoc(self, r1, r2, x1, x2, d, d2=None,sign1_arcsin=1.0,sign2_arcsin=1.0):
		N=50
		if d2==None:
			d2=d
		z1 = r1+x1  #circle center of anterior lens
		z2 = r2+x2  #circle center of posterior cornea
		dphi1=0.0
		dphi2=0.0
		if sign1_arcsin < 0:
			dphi1 = np.pi/2.0
		if sign2_arcsin < 0:
			dphi2 = np.pi/2.0
			
		phi1=np.linspace(0.0,np.absolute(np.arcsin(d/r1))+dphi1,N)
		phi2=np.linspace(np.absolute(np.arcsin(d2/r2))+dphi2,0.0,N)
		xc_1 = z1-r1*np.cos(phi1)
		yc_1 = r1*np.sin(phi1)
		xc_2 = z2-r2*np.cos(phi2)
		yc_2 = r2*np.sin(phi2)
		xc = np.append(xc_1,xc_2)
		yc = np.append(yc_1,yc_2)

		curve2d = []
		for (a,b) in zip(xc,yc):
			curve2d.append([a,b])
	
		mesh = self.revolve_curve(curve2d,axis="x", ang=2.0*np.pi, ang_pts=72)
		mesh.rotate(axis="y",angle=-np.pi/2.0,pivot = (0,0,0,0))
		return mesh

	def sphere(self,center,radius):
		N=72
		phi = np.linspace(0.0,2.0*np.pi,N)
		x = np.cos(phi)*radius
		y = np.sin(phi)*radius
		
		xy = []
		for (a,b) in zip(x,y):
			xy.append([a,b])
			
		mesh = self.revolve_curve(xy, axis="x", ang=np.pi, ang_pts=N+1)
		mesh.translate(center)
		return mesh
		
	def hemisphere(self,center,radius):
		N=72
		phi = np.linspace(0.0,np.pi,N)
		x = np.cos(phi)*radius
		y = np.sin(phi)*radius
		
		xy = []
		for (a,b) in zip(x,y):
			xy.append([a,b])
			
		mesh = self.revolve_curve(xy, axis="x", ang=np.pi, ang_pts=N+1)
		mesh.translate(center)
		return mesh
	
	def parabolic_mirror(self,focus=(0,0,0),focal_length=5.0,diameter=20.0,reflectivity = 0.98):
		N=72
		M=200
		yn = np.linspace(0.0,diameter/2.0,M)
		xn = yn**2/(4.0*focal_length) - focal_length
		
		x = focus[0] + xn
		y = focus[1] + yn
		
		xy = []
		for (a,b) in zip(x,y):
			xy.append([a,b])
			
		mesh = self.revolve_curve(xy, axis="x", ang=2.*np.pi, ang_pts=N+1)
		mesh.setMaterial(mat_type="mirror",reflectivity=reflectivity)
		return mesh

	def topless_cylinder(self,center=(0,0,0),diameter=20.0,height=10.0):
		N=72
		x = np.array([ 0.0, 0.0, 1.0]) * height   + center[1]
		y = np.array([ 0.0,  .5,  .5]) * diameter + center[0]
				
		xy = []
		for (a,b) in zip(x,y):
			xy.append([a,b])
			
		mesh = self.revolve_curve(xy, axis="x", ang=2.*np.pi, ang_pts=N+1)
		return mesh
		
	def revolve_curve(self, curve, axis="x", ang=2*np.pi, ang_pts=36):
		# curve		list of (x,y) tuples
		# axis		x or y axis
		# ang		angle to revolve (pi for half circle incase of symmetrical curve)
		# ang_pts	angular resolution
		
		Rx = lambda x: np.matrix([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
		Ry = lambda x: np.matrix([[np.cos(x),0,np.sin(x)],[0,1,0],[-np.sin(x),0,np.cos(x)]])
		Rz = lambda x: np.matrix([[np.cos(x),-np.sin(x),0],[np.sin(x),np.cos(x),0],[0,0,1]])
		
		if axis in {"x","X"}:
			R=Rx
		elif axis in {"y","Y"}:
			R=Ry
		elif axis in {"z","Z"}:
			R=Rz
		else:
			R=Rx
			
		verts=[]
		tris =[]
		
		curve3d=[]
		for xy in curve:
			curve3d.append([[xy[0]],[xy[1]],[0]])
		
		d_ang = ang / (ang_pts - 1.0)
		angs = np.linspace(0.0,ang-d_ang,ang_pts)
		k=0
		for (phi1,phi2) in zip(angs,angs+d_ang):
			for i in np.arange(len(curve3d)-1):
				v1 = R(phi1)*curve3d[i]
				v2 = R(phi1)*curve3d[i+1]
				v3 = R(phi2)*curve3d[i]
				v4 = R(phi2)*curve3d[i+1]
				verts.append(np.array([v1[0,0],v1[1,0],v1[2,0],0],dtype=np.float32))
				verts.append(np.array([v2[0,0],v2[1,0],v2[2,0],0],dtype=np.float32))
				verts.append(np.array([v3[0,0],v3[1,0],v3[2,0],0],dtype=np.float32))
				verts.append(np.array([v4[0,0],v4[1,0],v4[2,0],0],dtype=np.float32))
				tris.append([0,1,2]+i*4+4*k*(len(curve3d)-1))
				tris.append([2,3,1]+i*4+4*k*(len(curve3d)-1))
			k+=1
				
		return GeoObject(verts,tris)
			
	def lens_spherical_biconcave(self,focus,r1,r2,diameter,IOR):
		curve2d = self.lens_spherical_2r(focus,r1,r2,diameter,1,IOR)
		mesh = self.revolve_curve(curve2d, axis="x", ang=np.pi, ang_pts=36)
		mesh.setMaterial(mat_type="refractive",IOR=IOR)
		return mesh
		
	
	def curve_lens_spherical_biconcave(self,focus,r1,r2,d,diameter,axis,IOR):
		# focus         coordinate of focus
		# r1            radius left
		# r2            radius right
		# d             thickness of thinnes point of concave lens
		# diameter      lens diameter
		# axis		optical axis of lens
		# lens_type	concave or convex
		# n             refractive index of lens material

		abs=np.absolute
		n=IOR
		
		phi_r1 = np.arcsin(diameter/(2*r1))
		d1=abs(r1-r1*np.cos(phi_r1))
		f1=abs(1/((n-1)*(1/r1)))
		phi_r2 = np.arcsin(diameter/(2*r2))
		d2=abs(r2-r2*np.cos(phi_r2))
		f2=abs(1/((n-1)*(1/r2)))     

		f0=1/(1/f1+1/f2)
		d0=d
		
		fx0 = focus[0]
		fy0 = focus[1]

		(poly1,f1,d1)=self.curve_lens_spherical((fx0+f1+f0,  fy0),r1,diameter,-1,-1,IOR)
		(poly2,f2,d2)=self.curve_lens_spherical((fx0+d-f1+f0,fy0),r2,diameter, 1,-1,IOR)
		
		curve=poly1[0:-3]
		curve.extend(poly2[0:-3])
		curve.extend([poly1[1]])
		#print curve
		return (curve,f0,d0)

	def lens_spherical_2r(self,focus,r1,r2,diameter,lens_sign,n):
		N=60
		fx0 = focus[0]
		fy0 = focus[1]
		abs=np.absolute
		x=np.zeros(2*N+1)
		y=np.zeros(2*N+1)
		phi_r1 = np.arcsin(diameter/r1)
		phi_r2 = np.arcsin(diameter/r2)
		d=abs(r1-r1*np.cos(phi_r1))+abs(r2-r2*np.cos(phi_r2))
		f=abs(1/((n-1)*(1/r1-1/r2+(n-1)*d/(n*r1*r2))))
		q=(f-r1) # distance from f to center of r1 circle
		r1x0=fx0+q
		r1y0=fy0
		r2x0=fx0+q+r1-lens_sign*d+r2
		r2y0=fy0
		
		i=0
		for phi in np.linspace(np.pi-abs(phi_r2),np.pi+abs(phi_r2),N):
			x[i]=r2x0+r2*np.cos(phi)
			y[i]=r2y0+r2*np.sin(phi)
			i+=1

		for phi in np.linspace(-abs(phi_r1),abs(phi_r1),N):
			x[i]=r1x0+r1*np.cos(phi)
			y[i]=r1y0+r1*np.sin(phi)
			i+=1

		x[i]=x[0]
		y[i]=y[0]
		
		xy = []
		for (a,b) in zip(x,y):
			xy.append([a,b])
		
		return xy
		
	def curve_lens_spherical(self,focus,r1,diameter,lens_direction,lens_sign,IOR): #function [x,y,f,d]
		# fx0           focus coord x
		# fy0           focus coord y
		# r1            radius left
		# diameter      lens diameter
		# lens_direction     direction the lens curvature is facing
		# n             refractive index of lens material
		N=20
		n=IOR
		pi = np.pi
		
		abs=np.absolute
		x=np.zeros(N+3)
		y=np.zeros(N+3)
		phi_r1 = np.arcsin(diameter/(2*r1))
		d=abs(r1-r1*np.cos(phi_r1))
		f=abs(1/((n-1)*(1/r1)))
		q=f+r1 # distance from f to center of r1 circle
		 
		fx0 = focus[0]
		fy0 = focus[1]

		if lens_sign > 0:
			ang_offset = 0
		else:
			ang_offset = -pi


		if lens_direction >= 0:
			r1x0=fx0-lens_sign*q
			r1y0=fy0
			angles = np.linspace(ang_offset-lens_sign*abs(phi_r1),ang_offset+lens_sign*abs(phi_r1),N)
		else:
			r1x0=fx0+lens_sign*q
			r1y0=fy0
			angles = np.linspace(ang_offset+pi-lens_sign*abs(phi_r1),ang_offset+pi+lens_sign*abs(phi_r1),N)
		 
		i=0
		for phi in angles:
			x[i]=r1x0+r1*np.cos(phi)
			y[i]=r1y0+r1*np.sin(phi)
			i+=1
		 
		if lens_sign < 0:
			x[i]=x[i-1]+lens_direction*d
			y[i]=y[i-1]
			i+=1
			x[i]=x[i-2]+lens_direction*d
			y[i]=y[0]
			i+=1

		 
		x[i]=x[0]
		y[i]=y[0]
		 
		xy = []
		for (a,b) in zip(x,y):
			xy.append([a,b])
		 	
		return (xy,f,d)


if __name__ == '__main__':	
	print "generating test geometry"
	oe = optical_elements()
	m_l= oe.parabolic_mirror(focus=(0,0,0),focal_length=5.0,diameter=20.0,reflectivity = 0.98)  
	m_l.rotate(axis="y",angle=-np.pi/2,pivot = (0,0,0,0))
	m_l.write_dxf("./parabolic_mirror.dxf")

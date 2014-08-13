'''************************************************************************
    LightPyCL light source class
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
import numpy.linalg as la
from time import time
from dxfwrite import DXFEngine as dxf


class light_source():
	center = None #float3 coord
	direction = None #float3 normalized vector
	directivity = None #function of phi an thita measured from direction
	power = 1	#total power of light source (sum over all rays)
	ray_count = None #total ray count when asked for a set of rays
	rays_origin = None
	rays_dir   = None

	def __init__(self, center=np.array([0,0,0,0],dtype=np.float32), direction=(0,0,1), directivity=lambda x,y: np.cos(y), power=1, ray_count=500):
		self.center = center
		self.direction = direction
		self.directivity = directivity
		self.power = power
		self.ray_count = ray_count
		self.random_rays()

	def grid_rays(self):
		self.rays_origin = np.zeros((self.ray_count,4)).astype(np.float32)  + self.center
		self.rays_dest   = np.zeros((self.ray_count,4)).astype(np.float32)

		self.ray_count   = np.floor(np.sqrt(self.ray_count))**2
		print "Setting up rays in regular grid"
		print "Setting ray count to next square ray_count = ", self.ray_count
		#direction determines the main pointing direction of the beam => phi0 and thita0
		#use phi and thita for rotation matricies along x and z axis
		#first x for thita and then z for phi
		elevation0 = np.arccos(self.direction[2]/la.norm(self.direction))
		if self.direction[0] == 0:
			if self.direction[1] >= 0:
				azimuth0 = np.pi/2.0
			else:
				azimuth0 = -np.pi/2.0
		else:
			azimuth0 = np.arctan(self.direction[1]/self.direction[0])

		Rx = lambda x: np.matrix([[1,0,0,0],[0,np.cos(x),-np.sin(x),0],[0,np.sin(x),np.cos(x),0],[0,0,0,0]])
		Ry = lambda x: np.matrix([[np.cos(x),0,np.sin(x),0],[0,1,0,0],[-np.sin(x),0,np.cos(x),0],[0,0,0,0]])
		Rz = lambda x: np.matrix([[np.cos(x),-np.sin(x),0,0],[np.sin(x),np.cos(x),0,0],[0,0,1,0],[0,0,0,0]])
		
		elevation_ = np.transpose(np.matrix(np.linspace(0.0,np.pi/2.0,np.sqrt(self.ray_count)))) #elevation
		azimuth_   = np.transpose(np.matrix(np.linspace(0.0,2.0*np.pi,np.sqrt(self.ray_count))))  #azimuth
		
		for k in np.arange(np.sqrt(self.ray_count)):
			if k==0:
				elevation = elevation_
				azimuth   = azimuth_
			else:
				elevation = np.concatenate((elevation,elevation_),axis=0)
				azimuth   = np.concatenate((azimuth,0.0*azimuth_+np.float(azimuth_[k])),axis=0)
		
		sa = np.sin(azimuth)
		ca = np.cos(azimuth)
		se = np.sin(elevation)
		ce = np.cos(elevation)
	
		self.rays_power = np.array(self.directivity(azimuth,elevation),dtype=np.float32)*self.power
		#self.rays_power = self.rays_power / np.sum(self.rays_power)
		
		self.rays_dir = np.array(
		    np.append(np.append(np.append(
		    np.multiply(se,ca),np.multiply(se,sa),axis=1),
		    ce,axis=1),
		    np.zeros((self.ray_count,1)),axis=1),
		    dtype=np.float32)
		  
		self.rays_dir = np.dot(self.rays_dir,Rx(elevation0))
		self.rays_dir = np.dot(self.rays_dir,Rz(azimuth0))

	
	def random_rays(self):
		self.rays_origin = np.zeros((self.ray_count,4)).astype(np.float32)  + self.center
		self.rays_dest   = np.zeros((self.ray_count,4)).astype(np.float32)
		
		print "Setting up rays in random pattern"
		#direction determains the main pointing direction of the beam => phi0 and thita0
		#use phi and thita for rotation matricies along x and z axis
		#first x for thita and then z for phi
		elevation0 = np.arccos(self.direction[2]/la.norm(self.direction))
		if self.direction[0] == 0:
			if self.direction[1] >= 0:
				azimuth0 = np.pi/2.0
			else:
				azimuth0 = -np.pi/2.0
		else:
			azimuth0 = np.arctan(self.direction[1]/self.direction[0])

		Rx = lambda x: np.matrix([[1,0,0,0],[0,np.cos(x),-np.sin(x),0],[0,np.sin(x),np.cos(x),0],[0,0,0,0]])
		Ry = lambda x: np.matrix([[np.cos(x),0,np.sin(x),0],[0,1,0,0],[-np.sin(x),0,np.cos(x),0],[0,0,0,0]])
		Rz = lambda x: np.matrix([[np.cos(x),-np.sin(x),0,0],[np.sin(x),np.cos(x),0,0],[0,0,1,0],[0,0,0,0]])
		
		u = np.random.rand(self.ray_count,1)
		v = np.random.rand(self.ray_count,1)
		ar = np.zeros((self.ray_count,1))
		
		# sphere picking
		elevation = np.arccos(u) #only rays in +z are desired. otherwise use 2*u-1
		azimuth   = 2.0*np.pi*v

		xr = np.sin(elevation)*np.cos(azimuth)
		yr = np.sin(elevation)*np.sin(azimuth)
		zr = np.cos(elevation)
		rd = np.concatenate((xr,yr,zr,ar),axis=1)
		
		self.rays_dir = rd
		self.rays_power = np.float32(self.directivity(azimuth,elevation)) 
		self.rays_power = self.rays_power * self.power/np.sum(self.rays_power)
		  
		self.rays_dir = np.dot(self.rays_dir,Rx(elevation0))
		self.rays_dir = np.dot(self.rays_dir,Rz(azimuth0)).astype(np.float32)

		
	def rotate_rays(self,axis="z",pivot=[0,0,0,0],ang=np.pi/2.0):
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
		self.rays_origin = np.array(np.add(np.dot(np.subtract(self.rays_origin,np.array(pivot)),R(ang)),np.array(pivot))).astype(np.float32)
		self.rays_dir = np.array(np.dot(self.rays_dir,R(ang))).astype(np.float32)


	def save_dxf(self,dxf_file):
		drawing = dxf.drawing(dxf_file)
		drawing.add_layer('Rays', color=3)
		#drawing.add_layer('Geometry', color=5)
		
		print "Writing ryas to DXF file."
		for (r0,rd) in zip(self.rays_origin,self.rays_origin+self.rays_dir):
			#drawing.add(dxf.line(start=r0[0:3], end=rd[0:3],layer="Rays",thickness=1, linetype="SOLID"))
			drawing.add(dxf.face3d([r0[0:3], 
						rd[0:3], 
						rd[0:3]], layer="Rays"))


		drawing.save()	


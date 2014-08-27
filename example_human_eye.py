'''************************************************************************
    LightPyCL example: simulation of the human eye
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

# This example demonstrates
#	- the generation of meshes to resemble the optics of the human eye 
#	- setting up a colimated lightsource
#	- postprocessing and plotting the light distribution on the retina

import pyopencl as cl
import numpy as np
import pyopencl.array as cl_array
import numpy.linalg as la
from time import time
import geo_optical_elements as goe
from dxfwrite import DXFEngine as dxf
import light_source
import iterative_tracer as it
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

time1 = time()

# Amount of rays to trace
ray_count=100000
# tracer termination conditions. if either is reached, the tracer will stop tracing and return all results colledted until that point.
iterations = 16		#if this amount of iterations has been done, the tracer will stop.
power_dissipated = 0.99 # if 99% (0.99) amount of power has left the scene or was terminated/measured, the tracer will stop


# create one lightsource
ls0 = light_source.light_source(center=np.array([0,0,-10,0],dtype=np.float32), direction=(0,0.01,1), directivity=lambda x,y: 1.0+0.0*np.cos(y), power=1000., ray_count=ray_count)
ls0.random_collimated_rays(diameter=5.0)
# tracer code expects a list of light sources
ls = [ls0] 

# define the maximum length a ray can travel before being terminated
max_ray_len = np.float32(4.0e1)

# create an instance of optical elements so we can generate meshes for the scene
oe = goe.optical_elements()

# set IOR of environment
ior_env = np.float32(1.0)

# SETUP RADII, IOR AND DISTANCES FOR EYE
# data from: "Optics of the Human Eye" by W. N. Charman
r_cornea = 5.0 #cornea opening radius
r_lens   = 5.0 #lens opening radius

r_ac = 7.8    #anterior cornea radius
d_ac = 0.0    #anterior cornea position
r_pc = 6.5    #posterior cornea radius
d_pc = 0.55   #posterior cornea position
	
r_al = 10.2   #anterior lens radius
d_al = 3.6    #anterior lens position
r_pl = -6.0   #posterior lens radius
d_pl = 7.6    #posterior lens position

r_r = -12.1   #retina radius
d_r = 24.2   #retina position

IOR_c = 1.3771 #cornea IOR
IOR_l = 1.4200 #ior of lens
IOR_ah = 1.3374 #IOR of aqueus humour
IOR_vh = 1.336  #IOR of vitreous humour

retina = oe.hemisphere(center=[0,0,d_r/2.0,0],radius=-r_r*(1.0-1e-3))
retina.setMaterial(mat_type="measure")
meshes = [retina]

cornea = oe.spherical_lens_nofoc(r1=r_ac, r2=r_pc, x1=d_ac, x2=d_pc, d=r_cornea)
cornea.setMaterial(mat_type="refractive", IOR=IOR_c)
meshes.append(cornea)

lens = oe.spherical_lens_nofoc(r1=r_al, r2=r_pl, x1=d_al, x2=d_pl, d=r_lens)
lens.setMaterial(mat_type="refractive", IOR=IOR_l)
meshes.append(lens)

aqu_humour = oe.spherical_lens_nofoc(r1=r_pc, r2=r_al, x1=d_pc*(1.0+1e-6), x2=d_al*(1.0-1e-6), d=r_cornea, d2=r_lens)
aqu_humour.setMaterial(mat_type="refractive", IOR=IOR_ah)
meshes.append(aqu_humour)

vit_humour = oe.spherical_lens_nofoc(r1=r_pl, r2=r_r*(1.0+1e-6), x1=d_pl*(1.0+1e-6), x2=d_r, d=r_lens, d2=r_lens, sign2_arcsin=-1.0)
vit_humour.setMaterial(mat_type="refractive", IOR=IOR_vh)
meshes.append(vit_humour)

#meshes = [retina]


time2 = time()
prep_time = time2 - time1
	
# setup the tracer and select the CL platform and device to run the tracer on
# if platform_name or and device_name are incorrect then the first device of the first platform will be used
tr = it.CL_Tracer(platform_name="NVIDIA",device_name="460") #INIT properly

time1 = time()		
# run the iterative tracer
tr.iterative_tracer(light_source=ls,meshes=meshes,trace_iterations=iterations,trace_until_dissipated=power_dissipated,max_ray_len=max_ray_len,ior_env=ior_env)

time2 = time()
sim_time = time2 - time1

# fetch results for further processing
resulting_rays = tr.results

proc_ray_count=0
for res in resulting_rays:
	proc_ray_count += len(res[3])
	#print res[0],res[1],ls0.rays_dir
	
	
# fetch measured rays termination position and powers
measured_rays = tr.get_measured_rays()

# plot the data
#extent of measurement surface. ((xmin,xmax),(ymin,ymax))
#setting to +-pi/2 for hemisphere
m_surf_extent=((-np.pi/2.0,np.pi/2.0),(-np.pi/2.0,np.pi/2.0)) 
#spatial resolution of binning
m_points=90 
tr.plot_binned_data(limits=m_surf_extent,points=m_points,use_angular=True,use_3d=True)

# save traced scene to dxf file if the amount of rays is not too large.
# if the ray_count is too high the dxf file will be sea urchin porn.
if ray_count < 10000:
	tr.save_traced_scene("./eye.dxf")

# Show overall stats
print "Processed 	    ",proc_ray_count, "rays in ", sim_time, "s"
print "Measured  	    ", len(measured_rays[1]), "rays."
print "On average performed ", proc_ray_count*np.int64(tr.tri_count)/np.float64(sim_time), "RI/s"

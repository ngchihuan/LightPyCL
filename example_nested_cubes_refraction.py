'''************************************************************************
    LightPyCL example: directional pattern of a spherical lens
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
#	- the generation of optical elements and lightsources for the tracer
#	- demonstrates nested worst case placement with overlaping surfaces
#		NOTE: meshes with partially overlaping volumes should be avoided as the material assignment in the overlapping region is ambiguous. 
#			is n1 or n2 vaid in overlap? For the tracer this would then depend on impinging direction due to material hirarchy handling.
#	- demonstrates negative index material
#	- raytracer initialization and configuration
#	- postprocessing and plotting the results

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
ray_count=10
# Amount of iterations to run the tracer. defines the depth rays are followed to.
iterations = 6


# create one lightsource
ls0 = light_source.light_source(center=np.array([0,0,0,0],dtype=np.float32), direction=(0,0,1), directivity=lambda x,y: np.cos(y), power=1000., ray_count=ray_count)
#ls0.grid_rays()
# tracer code expects a list of light sources
ls = [ls0] 

# define the maximum length a ray can travel before being terminated
max_ray_len = np.float32(2.0e2)

# create an instance of optical elements so we can generate meshes for the scene
oe = goe.optical_elements()

# set IOR of environment
ior_env = np.float32(1.0)

# setup a hemisphere to measure the lightsources spatial power distribution
#measureSurf = oe.hemisphere(center=[0,0,0,0],radius=1000.0)
#measureSurf.setMaterial(mat_type="measure")
meshes = [] #[measureSurf]

m2 = oe.cube(center=(0,0,20,0),size=[100,100,10,0])
m2.setMaterial(mat_type="refractive",IOR=2.0)
meshes.append(m2)

m2 = oe.cube(center=(-20,0,22.5,0),size=[30,30,5,0])
m2.setMaterial(mat_type="refractive",IOR=3.0)
meshes.append(m2)

m2 = oe.cube(center=(20,0,20,0),size=[30,30,5,0])
m2.setMaterial(mat_type="refractive",IOR=-2.0)
meshes.append(m2)


time2 = time()
prep_time = time2 - time1
	
# setup the tracer and select the CL platform and device to run the tracer on
# if platform_name or and device_name are incorrect then the first device of the first platform will be used
tr = it.CL_Tracer(platform_name="NVIDIA",device_name="460") #INIT properly

time1 = time()		
# run the iterative tracer
tr.iterative_tracer(light_source=ls,meshes=meshes,trace_iterations=iterations,max_ray_len=max_ray_len,ior_env=ior_env)

time2 = time()
sim_time = time2 - time1

# fetch results for further processing
resulting_rays = tr.results

proc_ray_count=0
for res in resulting_rays:
	proc_ray_count += len(res[3])
	
# fetch measured rays termination position and powers
measured_rays = tr.get_measured_rays()

# plot the data
#extent of measurement surface. ((xmin,xmax),(ymin,ymax))
#setting to +-pi/2 for hemisphere
m_surf_extent=((-np.pi/2.0,np.pi/2.0),(-np.pi/2.0,np.pi/2.0)) 
#spatial resolution of binning
m_points=30 
#tr.plot_binned_data(limits=m_surf_extent,points=m_points,use_angular=True,use_3d=True)

# save traced scene to dxf file if the amount of rays is not too large.
# if the ray_count is too high the dxf file will be sea urchin porn.
if ray_count < 10000:
	tr.save_traced_scene("./results.dxf")

# Show overall stats
print "Processed 	    ",proc_ray_count, "rays in ", sim_time, "s"
print "Measured  	    ", len(measured_rays[1]), "rays."
print "On average performed ", proc_ray_count*np.int64(tr.tri_count)/np.float64(sim_time), "RI/s"

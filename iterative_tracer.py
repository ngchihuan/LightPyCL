'''************************************************************************
    LightPyCL iterative tracer code
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

import pyopencl as cl
import numpy as np
import pyopencl.array as cl_array
import numpy.linalg as la
from time import time, strftime
import geo_optical_elements as goe
from dxfwrite import DXFEngine as dxf
import pylab
import cPickle as pickle


class CL_Tracer():
	results = None
	geometry = None
	
	def __init__(self,platform_name="NVIDIA",device_name="770",debug=False):
		"""CL Tracer constructor 
			- initializes the CL platform and device
			- sets up the CL context and queue
			- loads and compiles the CL kernel  """
		## Step #1. Obtain an OpenCL platform.
		self.debug = debug
		platforms = cl.get_platforms()
		
		self.platform = cl.get_platforms()[0]
		for pf in platforms:
			if pf.name.find(platform_name) >= 0:
				self.platform = pf
		 		 
		## Step #2. Obtain a device id for at least one device (accelerator).
		devices = self.platform.get_devices()
		self.device = self.platform.get_devices()[0]
		for dev in devices:
			if dev.name.find(device_name) >= 0:
				self.device = dev
				
		print "Using CL device: ", self.platform.name, " - ", self.device.name

		## Step #3. Create a context for the selected device.
		self.context = cl.Context([self.device])
		 
		## Step #4. Create the accelerator program from source code.
		## Step #5. Build the program.
		## Step #6. Create one or more kernels from the program functions.
		print "Loading and building kernel."
		f=open("kernel_reflect_refract_intersect.cl","r")
		kernel_str = "".join(f.readlines())
		f.close()
		
		# find out how to used "-cl-nv-verbose" in order to see global/local/private mem usage stats
		self.prg = cl.Program(self.context, kernel_str).build()
		 
		## Step #7. Create a command queue for the target device.
		self.queue = cl.CommandQueue(self.context)
	

	def iterative_tracer(self,light_source,meshes,trace_iterations,max_ray_len,ior_env):
		""" iterative_tracer takes the light_source and meshes objects and coverts them to cl_arrays.
		    Thereafter it intersects and reflects and refracts all the rays from the light source(s).
		    Intersection, reflection and refraction are repeated trace_iterations number of times.
		    
		    New rays (reflected/refracted ones) are the result of every iteration and are used as the
		    input for the next iteration.
		    
		    Results are stored in the class property results as a list of tuples 
		    (rays_origin, rays_dest, power, measured) for every iteration.
		    
		    For very large numbers of input rays, rays are partitioned in order to prevent using more
		    memory than the CL device provides. The calculation of the maximum ray count is based on 
		    cl_device.global_mem_size and sizes of the buffers required for calculation and overhead."""
		    
		## Step #8. Allocate device memory and move input data from the host to the device memory.
		# rays
		max_ray_len = np.float32(max_ray_len)
		ior_env = np.float32(ior_env)
		print "Initializing variables and converting scene data."
				
		#prepare lightsource buffers
		k=0
		for light in light_source:
			if k==0:
				ray_count 	= np.int32(light.ray_count)
				rays_origin 	= np.float32(light.rays_origin)
				rays_dir 	= np.float32(light.rays_dir)
				rays_power 	= np.float32(light.rays_power)

			else:
				ray_count   = np.int32(ray_count + light.ray_count)
				rays_origin = np.append(rays_origin,light.rays_origin,axis=0).astype(np.float32)
				rays_dir    = np.append(rays_dir,light.rays_dir,axis=0).astype(np.float32)
				rays_power  = np.append(rays_power,light.rays_power,axis=0).astype(np.float32)

			k+=1

		rays_pow		= np.array(rays_power,dtype=np.float32) 
		rays_meas		= np.zeros(ray_count).astype(np.int32)
		rays_current_mid	= np.zeros(ray_count).astype(np.int32) - 2 #initial value -2 tell the tracer that the rays have been emitted and do not know in which (if any) mesh they are located
				
		#MESH does not change => only needs to be set up at start of tracer
		mesh_count=np.int32(len(meshes))
		mesh_mat_type 	= np.zeros(mesh_count,dtype=np.int32)
		mesh_ior	= np.zeros(mesh_count,dtype=np.float32)
		mesh_refl	= np.zeros(mesh_count,dtype=np.float32)
		mesh_diss	= np.zeros(mesh_count,dtype=np.float32)
		
		# convert meshes to linear buffer with mesh id for tracer to be able to iterate over
		j=0
		for mesh in meshes:	
			mesh_mat = mesh.getMaterialBuf() # np.float32(mesh.IOR)
						
			mesh_mat_type[j] = np.int32(mesh_mat.get("type"))
			mesh_ior[j]	 = np.float32(mesh_mat.get("IOR"))
			mesh_refl[j]	 = np.float32(mesh_mat.get("R"))
			mesh_diss[j]	 = np.float32(mesh_mat.get("dissipation"))
			
			tribuf = mesh.tribuf()
			m_id_tmp = np.zeros(len(tribuf[0]),dtype=np.int32)+j

			if j==0:
				m_v0 = np.array(tribuf[0],dtype=np.float32)
				m_v1 = np.array(tribuf[1],dtype=np.float32)
				m_v2 = np.array(tribuf[2],dtype=np.float32)
				m_id = np.array(m_id_tmp,dtype=np.int32)
			else:
				m_v0=np.append(m_v0,tribuf[0],axis=0).astype(np.float32)
				m_v1=np.append(m_v1,tribuf[1],axis=0).astype(np.float32)
				m_v2=np.append(m_v2,tribuf[2],axis=0).astype(np.float32)
				m_id=np.append(m_id,m_id_tmp,axis=0).astype(np.int32)
			j+=1
		tri_count = np.int32(len(m_v0))
		self.tri_count = tri_count
		print "Triangle count:              ", tri_count
		print "Mesh count:                  ", mesh_count

		#store geometry for other functions to access (save_results would be one such function)
		self.meshes = meshes
		self.geometry = (m_v0,m_v1,m_v2)
		
		#Geometry bufs
		m_v0_buf   = cl_array.to_device(self.queue,m_v0)
		m_v1_buf   = cl_array.to_device(self.queue,m_v1)
		m_v2_buf   = cl_array.to_device(self.queue,m_v2)
		m_id_buf   = cl_array.to_device(self.queue,m_id)
		
		m_typ_buf   = cl_array.to_device(self.queue,mesh_mat_type)
		m_ior_buf   = cl_array.to_device(self.queue,mesh_ior)
		m_refl_buf  = cl_array.to_device(self.queue,mesh_refl)
		m_diss_buf  = cl_array.to_device(self.queue,mesh_diss)
		
		# Get global memory size to estimate resource partitioning and calculate how many bytes are required for an additional ray.
		sFloat  = 4
		sFloat3 = sFloat * 4
		sInt    = 4
		global_mem_size = self.device.global_mem_size
		space_req_per_ray = sFloat3 * 7 + sFloat * 3 + sInt * 3 + sFloat * 3 + sFloat * mesh_count + 2 * sInt * mesh_count
		space_req_meshes  = tri_count * (3 * sFloat3 + sInt + sFloat)

		#estimated CL device memory overhead
		global_mem_overhead_est = global_mem_size * 0.15
		# number of rays that can fit into global memory when considering memory overhead and ray count dependent buffers
		# currently limited max_ray_count to 50000 because larger values seem to hang my graphics cards sometimes. this does not seem to impact performance much though.
		max_ray_count 		= 50000 #np.int32(np.floor((global_mem_size - space_req_meshes - global_mem_overhead_est) / space_req_per_ray)) 
		
		print "Available space on CL dev:   ", global_mem_size / 1024**2, " MB"
		print "Space required per ray:      ", space_req_per_ray, " Bytes"
		print "Space required for all rays: ", space_req_per_ray * ray_count / 1024**2, " MB"
		print "Space required for geometry: ", space_req_meshes / 1024**2, " MB"
		print "Total space required:        ", (space_req_meshes + space_req_per_ray * ray_count) / 1024**2, " MB"
		print "Maximum permitted ray count: ", max_ray_count
		
		
		# SET UP CL DEVICE TEMP CALCULATION AND RESULT BUFFERS
		
		# Partition input rays depending on resources of CL device
		# setup partitioning variables
		
		# internal buffers! they must be set up ahead of calculation because dynamic size is not allowed
		# set up buffers that don't need initializing. they will be used by the kernels as they see fit and saves setup time.
		# if ray_count is smaller than only that number of rays will be processed by setting the global id size
		# if ray_count is larger than the maximum number of rays that can fit into global memory then do partitioning
		
		# TODO: find cleverer way to do partitioning. currently part_ray_count = ray_count if ray_count is smaller than max_ray_count.
		#	this automatically leads to partitioning when more than ray_count new rays are generated by reflection and refraction.
		#	however, partitioning is not necessary if max_ray_count is not reached. the complication here is CL_Array creation.
		#	Either time is lost by partitioning or buffer creation in every iteration. which gives more performance?
		part_ray_count  = max(min(ray_count,max_ray_count),50000)
			
		r_dest_buf 	= cl_array.zeros(self.queue,(part_ray_count,4),dtype=np.float32) 
		rr_origin_buf	= cl_array.zeros(self.queue,(part_ray_count,4),dtype=np.float32) 
		rr_dir_buf	= cl_array.zeros(self.queue,(part_ray_count,4),dtype=np.float32) 
		rr_pow_buf	= cl_array.zeros(self.queue,(part_ray_count,1),dtype=np.float32) 
		rr_meas_buf	= cl_array.zeros(self.queue,(part_ray_count,1),dtype=np.int32) 
		rt_origin_buf	= cl_array.zeros(self.queue,(part_ray_count,4),dtype=np.float32) 
		rt_dir_buf	= cl_array.zeros(self.queue,(part_ray_count,4),dtype=np.float32) 
		rt_pow_buf	= cl_array.zeros(self.queue,(part_ray_count,1),dtype=np.float32) 
		rt_meas_buf	= cl_array.zeros(self.queue,(part_ray_count,1),dtype=np.int32) 
			
		r_entering_buf 		= cl_array.zeros(self.queue,(part_ray_count,1),dtype=np.int32) 
		r_isect_m_id_buf	= cl_array.zeros(self.queue,(part_ray_count,1),dtype=np.int32) 
		r_isect_m_idx_buf 	= cl_array.zeros(self.queue,(part_ray_count,1),dtype=np.int32) 
		r_n1_m_id_buf 		= cl_array.zeros(self.queue,(part_ray_count,1),dtype=np.int32) 
		r_n2_m_id_buf 		= cl_array.zeros(self.queue,(part_ray_count,1),dtype=np.int32) 

		isects_count_buf 		= cl_array.zeros(self.queue,(part_ray_count,mesh_count),dtype=np.int32) 
		ray_isect_mesh_idx_tmp_buf 	= cl_array.zeros(self.queue,(part_ray_count,mesh_count),dtype=np.int32) 
				
		# create results variable and iterate
		self.results = []		
		for t_iter in np.arange(trace_iterations):
			print " "
			print "ITERATION", t_iter+1, "with ", ray_count, "rays."
			print "================================================================"

			NParts 	= np.int32(np.ceil(np.float32(ray_count)/np.float32(part_ray_count))) # amount of partitioning required to process all rays
			if NParts > 1:
				print "Too many rays for single pass. Partitioning rays into ", NParts, " groups."
	
			# setup result buffers			
			rays_dest		= np.empty_like(rays_origin).astype(np.float32)
			rrays_origin 		= np.empty_like(rays_origin).astype(np.float32)
			rrays_dir 		= np.empty_like(rays_origin).astype(np.float32)
			rrays_meas		= np.empty_like(rays_meas).astype(np.float32)
			rrays_pow		= np.empty_like(rays_meas).astype(np.float32)
			trays_origin		= np.empty_like(rays_origin).astype(np.float32)
			trays_dir		= np.empty_like(rays_origin).astype(np.float32)
			trays_meas		= np.empty_like(rays_meas).astype(np.float32)
			trays_pow		= np.empty_like(rays_meas).astype(np.float32)
		
			for part in np.arange(0,NParts,1):
				if NParts > 1:
					print ""
					print "ITERATION", t_iter+1, "partition", part+1, "of", NParts, "ray partitions."
					print "----------------------------------------------------------------"
					
				isect_min_ray_len_buf	= cl_array.zeros(self.queue,(part_ray_count,mesh_count),dtype=np.float32) + max_ray_len
				
				# partitioning indices
				minidx = part * part_ray_count
				maxidx = min((part + 1) * part_ray_count - 1, ray_count - 1) + 1 #+1 because array[x:y] accesses elements x through y-1
				
				part_ray_count_this_iter = maxidx - minidx #rays that will be processed in this iteration
				
				# CREATE DEVICE INPUT DATA
				print "Seting us up the buffers on the CL device."
				mf = cl.mem_flags
				time1 = time()
				# ray bufs
				r_origin_buf	 	= cl_array.to_device(self.queue,rays_origin[minidx:maxidx,:])	#needs copy
				r_dir_buf 		= cl_array.to_device(self.queue,rays_dir[minidx:maxidx,:])	#needs copy
				r_pow_buf 		= cl_array.to_device(self.queue,rays_pow[minidx:maxidx])	#needs copy
				r_meas_buf		= cl_array.to_device(self.queue,rays_meas[minidx:maxidx])	#needs copy
				r_prev_isect_m_id_buf   = cl_array.to_device(self.queue,rays_current_mid[minidx:maxidx])	#needs copy
				#INTERSECT RAYS WITH SCENE
				GIDs = (part_ray_count_this_iter,1) # max global ids set to number of input rays
				print "Starting intersect parallel ray kernel."
				event = self.prg.intersect(self.queue, GIDs, None, r_origin_buf.data,
					r_dir_buf.data,r_dest_buf.data,r_entering_buf.data,
					r_isect_m_id_buf.data,r_isect_m_idx_buf.data,
					m_v0_buf.data,m_v1_buf.data,m_v2_buf.data,m_id_buf.data,
					isect_min_ray_len_buf.data,isects_count_buf.data,
					ray_isect_mesh_idx_tmp_buf.data,np.int32(mesh_count),
					np.int32(tri_count),np.int32(ray_count),np.float32(max_ray_len))
				event.wait()
				time2 = time()
				t_intersect = time2 -time1
				print "Intersection execution time:   ", time2 - time1, "s"
			
				#POSTPROCESS INTERSECT RESULTS
				print "Running intersect postprocessing kernel."
				time1 = time()
				event2 = self.prg.intersect_postproc(self.queue, GIDs, None, r_origin_buf.data,
					r_dir_buf.data,r_dest_buf.data, r_prev_isect_m_id_buf.data, 
					r_n1_m_id_buf.data, r_n2_m_id_buf.data, r_entering_buf.data,
					r_isect_m_id_buf.data,r_isect_m_idx_buf.data,
					m_v0_buf.data,m_v1_buf.data,m_v2_buf.data,m_id_buf.data,m_typ_buf.data,
					isect_min_ray_len_buf.data,isects_count_buf.data,
					ray_isect_mesh_idx_tmp_buf.data,mesh_count,ray_count,max_ray_len)
				event2.wait()
				time2 = time()
				t_postproc = time2 -time1
				print "Intersect postprocessing time: ", time2 - time1, "s" 

				# REFLECT AND REFRACT INTERSECTED RAYS
				print "Running Fresnell kernel."
				time1 = time()
				event3 = self.prg.reflect_refract_rays(self.queue, GIDs, None, r_origin_buf.data, r_dest_buf.data,
					r_dir_buf.data, r_pow_buf.data, r_meas_buf.data, r_entering_buf.data, 
					r_n1_m_id_buf.data, r_n2_m_id_buf.data,
					rr_origin_buf.data, rr_dir_buf.data, rr_pow_buf.data, rr_meas_buf.data,
					rt_origin_buf.data, rt_dir_buf.data, rt_pow_buf.data, rt_meas_buf.data,
					r_isect_m_id_buf.data, r_isect_m_idx_buf.data, m_v0_buf.data, m_v1_buf.data,
					m_v2_buf.data, m_id_buf.data, m_typ_buf.data, m_ior_buf.data, m_refl_buf.data,
					m_diss_buf.data , ior_env, mesh_count, ray_count, max_ray_len)
				event3.wait()
				time2 = time()
				t_fresnell = time2 -time1
				print "Fresnell processing time:      ", time2 - time1, "s" 
				print "Performance:                   ", (np.int64(part_ray_count_this_iter)*np.int64(tri_count))/np.float128(t_fresnell+t_intersect+t_postproc), "RI/s"

				# FETCH RESULTS FROM CL DEV
				print "Fetching results from device."
				time1 = time()
				rrays_origin[minidx:maxidx,:] 		= rr_origin_buf.get(self.queue)[0:part_ray_count_this_iter,:]
				rrays_dir[minidx:maxidx,:] 		= rr_dir_buf.get(self.queue)[0:part_ray_count_this_iter,:]
				rrays_meas[minidx:maxidx]		= rr_meas_buf.get(self.queue).flatten()[0:part_ray_count_this_iter]
				rrays_pow[minidx:maxidx]		= rr_pow_buf.get(self.queue).flatten()[0:part_ray_count_this_iter]
				trays_origin[minidx:maxidx,:]		= rt_origin_buf.get(self.queue)[0:part_ray_count_this_iter,:]
				trays_dir[minidx:maxidx,:]		= rt_dir_buf.get(self.queue)[0:part_ray_count_this_iter,:]
				trays_meas[minidx:maxidx]		= rt_meas_buf.get(self.queue).flatten()[0:part_ray_count_this_iter]
				trays_pow[minidx:maxidx]		= rt_pow_buf.get(self.queue).flatten()[0:part_ray_count_this_iter]
				
				rays_current_mid[minidx:maxidx] 	= r_isect_m_id_buf.get(self.queue).flatten()[0:part_ray_count_this_iter]
				
				rays_dest[minidx:maxidx,:] 	= r_dest_buf.get(self.queue)[0:part_ray_count_this_iter,:]
				rays_pow[minidx:maxidx]  	= r_pow_buf.get(self.queue)[0:part_ray_count_this_iter]
				rays_meas[minidx:maxidx] 	= r_meas_buf.get(self.queue).flatten()[0:part_ray_count_this_iter]
				
				time2 = time()
				print "Fetching results took          ", time2 - time1, "s."
				
			# APPEND RESULTS OF THIS ITERATION TO OVERALL RESULTS 
			print "Assembling results"
			self.results.append((rays_origin,rays_dest,rays_pow,rays_meas))

			#print rays_origin
			#print rays_dest
			#print rays_meas
			
			# ASSEMBLE INPUT RAYS FOR NEXT ITERATION OR QUIT LOOP IF NO RAYS ARE LEFT
			print "Setting up for next iter."
		
			# remove measured and terminated rays for next iteration
			time1 = time()
			idx 	    = np.where(np.concatenate((rrays_meas,trays_meas),axis=0)==0)[0] #filter index for unmeasured/unterminated result rays
			rays_origin = np.append(rrays_origin,trays_origin,axis=0).astype(np.float32)[idx]
			rays_dir    = np.append(rrays_dir,trays_dir,axis=0).astype(np.float32)[idx]
			rays_pow    = np.append(rrays_pow,trays_pow,axis=0).astype(np.float32)[idx]
			rays_meas   = np.append(rrays_meas,trays_meas,axis=0).astype(np.int32)[idx]

			rays_current_mid   = np.append(rays_current_mid,rays_current_mid,axis=0).astype(np.int32)[idx]
			
			ray_count   = np.int32(len(rays_origin))
		
			if ray_count == 0:
				print "No rays left to trace. Terminating tracer."
				break
			time2 = time()
			t_host_pproc = time2 -time1
			print "Host side data pruning:        ", t_host_pproc, "s"
				
		return self.results
	
	def get_measured_rays(self):
		""" From all the results filter out only those that hit the measurement surface and return their endpoints and powers in the tuple (pos,pwr). """
		pos = None
		pwr = None
		k   = 0
		for (rays_origin,rays_dest,r_pow,r_meas) in self.results:
			idx         = np.where(r_meas>=.9)[0]
			rays_dest_m = rays_dest[idx]
			r_pow_m	    = r_pow[idx]
			if k==0:
				pos = rays_dest_m
				pwr = r_pow_m
			else:
				pos =  np.concatenate((pos,rays_dest_m),axis=0)
				pwr =  np.concatenate((pwr.flatten(),r_pow_m.flatten()),    axis=0)
			k+=1
		return (pos,pwr)
	
	def get_binned_data(self,limits=((-10,10),(-10,10)),points=500):
		"""  bin data for visualisation with points number of points within limits=((xmin,xmax),(ymin,ymax))."""
		(pos,pwr) = self.get_measured_rays()
		(H,x_coord,y_coord)=np.histogram2d(x=pos[:,0],y=pos[:,1],bins=points,range=limits,weights=pwr)
		self.hist_data = (H,x_coord,y_coord)
		return self.hist_data
		
	def plot_elevation_histogram(self,points=500,pole=[0,0,1,0]): 
		""" collect only elevation of all rays and plot histogram of them."""
		(pos,pwr) = self.get_measured_rays()
		
		pos0 = np.array(np.divide(pos,np.matrix(np.linalg.norm(pos,axis=1)).T))
		pwr = np.float64(pwr).flatten()
		elevation = np.arccos(np.dot(pos0,pole)).flatten()
		
		#calculate 1D histogram over all elevations
		import matplotlib.pyplot as plt
		#idx = np.where(elevation > 1e-6)[0]
		#(H,x)=np.histogram(elevation[idx],bins=points,weights=(pwr[idx].T/np.sin(elevation[idx])).flatten())
		(H,x)=np.histogram(elevation,bins=points,weights=pwr)
		x = (x[0:-1] + x[1:])/2.0
		dx = x[1] - x[0]
		H = H/(np.sin(x)*dx)
		plt.plot(x*180.0/np.pi,H)
		plt.title("Elevation Histogram")
		plt.xlabel("Elevation (Deg)")
		plt.ylabel("Intensity")
		pylab.savefig("./elevation_power_distribution.pdf")
		plt.show()

	def get_beam_width_half_power(self,points=500,pole=[0,0,1,0]): 
		""" collect only elevation of all rays and plot histogram of them."""
		(pos,pwr0) = self.get_measured_rays()
		
		pos0 = np.array(np.divide(pos,np.matrix(np.linalg.norm(pos,axis=1)).T))
		pwr0 = np.float64(pwr0)
		elevation0 = np.arccos(np.dot(pos0,pole))
		SIDX = lambda s: sorted(range(len(s)), key=lambda k: s[k])
		sort_idx = SIDX(elevation0)
		elevation = elevation0[sort_idx]
		pwr = pwr0[sort_idx]
		pwr_cumsum = np.cumsum(pwr)
		pwr_hmax = pwr_cumsum[-1]/2.0
		pwr_hm_idx = np.where(np.absolute(pwr_cumsum-pwr_hmax)==min(np.absolute(pwr_cumsum-pwr_hmax)))
		
		elevation_hmax = elevation[pwr_hm_idx]
		pwr_sum_hmax = pwr_cumsum[pwr_hm_idx]
		
		print ""
		print "Throughput results:"
		print "==================="
		print "Total measured power: ", pwr_cumsum[-1] #every ray carries an amount of power (Energy/s). Therefore sum over all rays with power gives overall power.
		print "Beam halfpower angle: ", elevation_hmax/np.pi*180.0, "Deg"
		print "Halfwidth throughput: ", pwr_sum_hmax/pwr_cumsum[-1]*100.0, "%"
		
	def get_beam_HWHM(self,points=500,pole=[0,0,1,0]): 
		""" collect only elevation of all rays and plot histogram of them."""
		(pos,pwr0) = self.get_measured_rays()
		
		pos0 = np.array(np.divide(pos,np.matrix(np.linalg.norm(pos,axis=1)).T))
		pwr0 = np.float64(pwr0)
		elevation0 = np.arccos(np.dot(pos0,pole))
		elevation1 = elevation0
		#idx = np.where(elevation0>1e-6)
		elevation0  = elevation0#[idx] 
		intensity0  = pwr0 #(pwr0[idx].T/np.sin(elevation0)).flatten()
		
		SIDX = lambda s: sorted(range(len(s)), key=lambda k: s[k])
		sort_idx = SIDX(elevation0)
		
		elevation = elevation0[sort_idx].flatten()
		intensity = intensity0[sort_idx].flatten()
		
		(H,x) = np.histogram(a=elevation,bins=points,weights=intensity)
		x = (x[0:-1] + x[1:])/2.0
		H = (H.T/np.sin(x)).flatten()
		
		pwr_hmax = max(H)/2.0
		pwr_hm_idx = np.where(np.absolute(H-pwr_hmax)==min(np.absolute(H-pwr_hmax)))[0]
		
		elevation_hmax = x[pwr_hm_idx]
		pwr_sum_hmax = np.sum(pwr0[np.where(elevation1 < elevation_hmax)])
		
		print ""
		print "Throughput results HWHM:"
		print "========================"
		print "Total measured power: ", sum(pwr0) #every ray carries an amount of power (Energy/s). Therefore sum over all rays with power gives overall power.
		print "Beam HWHM angle:      ", elevation_hmax/np.pi*180.0, "Deg"
		print "HWHM throughput:      ", pwr_sum_hmax/sum(pwr0)*100.0, "%"
		
	def get_binned_data_stereographic(self,limits=((-1,1),(-1,1)),points=500): #project data stereographically onto xy plane and bin it
		""" stereographically project measured ray endpoints and bin them on the CL DEV. This is a lot faster when you have loads of data. Binning is done with points number of points within limits=((xmin,xmax),(ymin,ymax))."""
		(pos0,pwr0) = self.get_measured_rays()
		pos0_dev = cl_array.to_device(self.queue,pos0.astype(np.float32))
		x_dev	 = cl_array.zeros(self.queue,pwr0.shape,dtype=np.float32)
		y_dev	 = cl_array.zeros(self.queue,pwr0.shape,dtype=np.float32)
		pwr0_dev = cl_array.to_device(self.queue,pwr0.astype(np.float32))
		pwr_dev  = cl_array.zeros(self.queue,pwr0.shape,dtype=np.float32)
		pivot    = cl_array.to_device(self.queue,np.array([0,0,0,0],dtype=np.float32))
			
		time1 = time()
		R_dev = cl_array.to_device(self.queue,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]]).astype(np.float32))
		evt = self.prg.stereograph_project(self.queue, pwr0.shape, None, pos0_dev.data,pwr0_dev.data,R_dev.data,pivot.data,x_dev.data,y_dev.data,pwr_dev.data)
			
			
		evt.wait()
			
		x=x_dev.get()
		y=y_dev.get()
		pwr=np.float64(pwr_dev.get())
	
		time2 = time()
		dx = np.float64(limits[0][1]-limits[0][0])/np.float64(points)
		dy = np.float64(limits[1][1]-limits[1][0])/np.float64(points)
		pwr = pwr / (dx * dy)
		
		(H,x_coord,y_coord)=np.histogram2d(x=x.flatten(),y=y.flatten(),bins=points,range=limits,weights=pwr.flatten())
		self.hist_data = (H,x_coord,y_coord)
		return self.hist_data


	def get_binned_data_angular(self,limits=((-1,1),(-1,1)),points=500):
		""" Azimuth/elevation map measured ray endpoints to a circle and bin them on the CL DEV. This linearly maps elevation to the circle's radius and azimuth to phi. nice for cross-section plots of directivity. Binning is done with points number of points within limits=((xmin,xmax),(ymin,ymax))."""
		(pos0,pwr0) = self.get_measured_rays()
		pos0_dev = cl_array.to_device(self.queue,pos0.astype(np.float32))
		x_dev	 = cl_array.zeros(self.queue,pwr0.shape,dtype=np.float32)
		y_dev	 = cl_array.zeros(self.queue,pwr0.shape,dtype=np.float32)
		pwr0_dev = cl_array.to_device(self.queue,pwr0.astype(np.float32))
		pwr_dev  = cl_array.zeros(self.queue,pwr0.shape,dtype=np.float32)
		pivot    = cl_array.to_device(self.queue,np.array([0,0,0,0],dtype=np.float32))
			
		time1 = time()
		R_dev = cl_array.to_device(self.queue,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]]).astype(np.float32))
		evt = self.prg.angular_project(self.queue, pwr0.shape, None, pos0_dev.data,pwr0_dev.data,R_dev.data,pivot.data,x_dev.data,y_dev.data,pwr_dev.data)
			
			
		evt.wait()
			
		x=x_dev.get()
		y=y_dev.get()
		pwr=np.float64(pwr_dev.get())
	
		time2 = time()
		dx = np.float64(limits[0][1]-limits[0][0])/np.float64(points)
		dy = np.float64(limits[1][1]-limits[1][0])/np.float64(points)
		pwr = pwr / (dx * dy)
		
		(H,x_coord,y_coord)=np.histogram2d(x=x.flatten(),y=y.flatten(),bins=points,range=limits,weights=pwr.flatten())
		self.hist_data = (H,x_coord,y_coord)
		return self.hist_data

	def replicate_lightsources_and_plot(self,limits=((-10,10),(-10,10)),points=500,axis="z",sources=36,use_3D=True):
		""" replicate_lightsources_and_plot use rotational symmetry of the scene to emulate an arbitrary number of lightsources with the traced results of one source. It emulates sources number of sources by rotating the resulting rays around the axis of rotational symmetry which is given by axis."""
		# set up rotation matrices
		Rx = lambda x: np.matrix([[1,0,0,0],[0,np.cos(x),-np.sin(x),0],[0,np.sin(x),np.cos(x),0],[0,0,0,0]])
		Ry = lambda x: np.matrix([[np.cos(x),0,np.sin(x),0],[0,1,0,0],[-np.sin(x),0,np.cos(x),0],[0,0,0,0]])
		Rz = lambda x: np.matrix([[np.cos(x),-np.sin(x),0,0],[np.sin(x),np.cos(x),0,0],[0,0,1,0],[0,0,0,0]])
		
		# user selection of rotation axis
		if axis in {"x","X"}:
			R=Rx
		elif axis in {"y","Y"}:
			R=Ry
		elif axis in {"z","Z"}:
			R=Rz
		else:
			R=Rx
		
		# fetch measured results
		(pos0,pwr0) = self.get_measured_rays()
		print ""
		print "Rotating measured rays to simulate ", sources, " light sources."
		print "----------------------------------------------------------------"

		# send data to CL dev for faster processing. also initialize result buffers.
		pos0_dev = cl_array.to_device(self.queue,pos0.astype(np.float32))
		x_dev	 = cl_array.zeros(self.queue,pwr0.shape,dtype=np.float32)
		y_dev	 = cl_array.zeros(self.queue,pwr0.shape,dtype=np.float32)
		pwr0_dev = cl_array.to_device(self.queue,pwr0.astype(np.float32))
		pwr_dev  = cl_array.zeros(self.queue,pwr0.shape,dtype=np.float32)
		pivot    = cl_array.to_device(self.queue,np.array([0,0,0,0],dtype=np.float32))
			
		time1 = time() # performance stats gathering
		
		# Replicate sources and perform angular mapping for each emulated source on CL Dev
		for k in np.arange(sources):
			# Only rotation matrix needs to be updated on CL dev. everything else is the same. just set R and fetch data from dev :)
			ang = k * 2.0 * np.pi/sources
			R_dev = cl_array.to_device(self.queue,R(ang).astype(np.float32))
			evt = self.prg.angular_project(self.queue, pwr0.shape, None, pos0_dev.data,pwr0_dev.data,R_dev.data,pivot.data,x_dev.data,y_dev.data,pwr_dev.data)
				
				
			evt.wait()
			
			# sum up results
			if k==0:
				x=x_dev.get()
				y=y_dev.get()
				pwr=np.float64(pwr_dev.get())
			else:
				x =  np.concatenate((x,x_dev.get()),axis=0)
				y =  np.concatenate((y,y_dev.get()),axis=0)
				pwr =  np.concatenate((pwr,np.float64(pwr_dev.get())),axis=0)
		time2 = time()
		print "Rotating results:  ", time2-time1, "s"
	
		print "Binning Data and plotting ... "
		#stereographic projection
		time1 = time()
		dx = np.float64(limits[0][1]-limits[0][0])/np.float64(points)
		dy = np.float64(limits[1][1]-limits[1][0])/np.float64(points)
		pwr = pwr /(dx*dy) # final surface element scaling

		# perform actual binning
		(H,x_coord,y_coord)=np.histogram2d(x=x.flatten(),y=y.flatten(),bins=points,range=limits,weights=pwr.flatten())
		self.hist_data = (H,x_coord,y_coord)
		
		xedges = x_coord * 180.0 / np.pi
		yedges = y_coord * 180.0 / np.pi
		extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
		time2 = time()
		print "Binning data:      ", time2-time1, "s"
		
		
		# How do you want to look at your data? 2D or 3D?
		if use_3D:
			import matplotlib.pyplot as plt
			from mpl_toolkits.mplot3d import Axes3D
			from matplotlib import cm
			from matplotlib.ticker import LinearLocator, FormatStrFormatter
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			X, Y = np.meshgrid(xedges[0:-1], yedges[0:-1])
			surf = ax.plot_surface(X, Y, H, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        		#ax.set_zlim(-1,1)
        		ax.zaxis.set_major_locator(LinearLocator(10))
			ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        		fig.colorbar(surf, shrink=0.5, aspect=5)
        		pylab.savefig("./binned_3D_data_replicated_sources.pdf")
			plt.show()
		else: #if you really want 2D, have it!
			pylab.imshow(H,extent=extent,origin='lower')
			pylab.colorbar()
			pylab.savefig("./binned_2D_data_replicated_sources.pdf")
			pylab.show()
		
		
	# plot binned data
	def plot_binned_data(self,limits=((-10,10),(-10,10)),points=500,use_3d=True,use_angular=False,hist_data =None):
		if hist_data == None:
			if use_angular:
				self.get_binned_data_angular(limits=limits,points=points)
				efact = 1.0
			else:
				(pos,pwr) = self.get_measured_rays()
				(H,x_coord,y_coord)=np.histogram2d(x=np.array(pos[:,0].flatten()),
								   y=np.array(pos[:,1].flatten()),
								   bins=points,range=limits,normed=False,
								   weights=np.array(np.float64(pwr).flatten()))
				self.hist_data = (H.astype(np.float64),x_coord,y_coord)
				efact = 1.0
				
			H = self.hist_data[0]
			xedges = self.hist_data[1] * efact
			yedges = self.hist_data[2] * efact
		else:
			if use_angular:
				efact = 90.0
			else:
				efact = 1.0
				
			self.hist_data = hist_data
			H = self.hist_data[0]
			xedges = self.hist_data[1]
			yedges = self.hist_data[2]	
				
		extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
		
		if use_3d:
			import matplotlib.pyplot as plt
			from mpl_toolkits.mplot3d import Axes3D
			from matplotlib import cm
			from matplotlib.ticker import LinearLocator, FormatStrFormatter
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			X, Y = np.meshgrid(xedges[0:-1], yedges[0:-1])
			surf = ax.plot_surface(X, Y, H, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        		ax.zaxis.set_major_locator(LinearLocator(10))
			ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        		fig.colorbar(surf, shrink=0.5, aspect=5)
        		pylab.savefig("./binned_3D_data.pdf")
			plt.show()
		else:
			pylab.imshow(np.log10(H),extent=extent,interpolation='nearest',origin='lower')
			pylab.colorbar()
			pylab.savefig("./binned_2D_data.pdf")
			pylab.show()
		
	def pickle_results(self):
		try:
			res_str  = pickle.dumps((self.results,self.meshes),1)
			timestring = strftime("%Y.%m.%d.%H.%M.%S")
			print timestring
			fname = "./{0}-tracer_results.txt".format(timestring)
		
			# open file
			f     = open(fname,"wb")
		
			# write data (overwrite is default)
			f.write(res_str)
		
			# close files
			f.close()
			
		except:
			print "Pickling results failed."
			f.close()
			return None
		else:
			print "Pickled results and meshes to ", fname
			return fname
			
	def load_pickle_results(self,path):
		try:
			# open file
			f     = open(path,"rb")
		
			# read data
			res_str = f.read()
			# close files
			f.close()

			(self.results,self.meshes)  = pickle.loads(res_str)

		except:
			print "UnPickling results failed."
			f.close()
		else:
			print "UnPickled results and meshes from ", path

		
	def save_traced_scene(self,dxf_file):
		""" write geometry and traced rays to a DXF file for your viewing pleasure. It's no fun with more than a few 100 rays though. 
		 >1000 rays look like sonic the sea urchin."""
		drawing = dxf.drawing(dxf_file)
		drawing.add_layer('Rays', color=3)
		drawing.add_layer('Geometry', color=5)
		
		print "Writing results as DXF file."
		for res in self.results:
			rays_origin	= res[0]
			rays_dest	= res[1]
		
			#draw rays to dxf
			for (r0,rd) in zip(rays_origin,rays_dest):
				drawing.add(dxf.face3d([r0[0:3], 
							rd[0:3], 
							rd[0:3]], layer="Rays"))

		#draw facets
		(m_v0,m_v1,m_v2)=self.geometry
		for (t0,t1,t2) in zip(m_v0,m_v1,m_v2):
			drawing.add(dxf.face3d([t0[0:3], 
						t1[0:3], 
						t2[0:3]], layer="Geometry"))

		drawing.save()	


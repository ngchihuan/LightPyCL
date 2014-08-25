/************************************************************************
    LightPyCL kernel code
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
**************************************************************************/

//Notes:
// - use one kernel to calculate intersections and generate new rays
// - call from python for each iteration => geometry remains in GPU memory, rays are trimmed
//   => worst case rays need to be partitioned when they become too many.
//	=> estimation for 9 bounces of 10k rays = 1024 * 10krays = 10Mrays => currently ~1GB VRAM required
//   => iteration count remains low (few read/write cycles)
// - kernel for postprocessing (calculate angles for each measured ray)
//	or binning kernel (sorting/summing)

/**
mesh_count 		= 4
ray_count		= 1e4
iterations		= 9 => 1024 ray_multiplyer

rays_origin 		= 4xfloat32 = 16
rays_dir 		= 16
rays_dest		= 16
ray_entering		= 4
ray_isect_mesh_id	= 4
ray_isect_mesh_idx	= 4
isect_min_ray_len	= 4 * mesh_count = 16
isects_count		= 4 * mesh_count = 16
ray_isect_mesh_idx_tmp	= 4 * mesh_count = 16
TOTAL			= 108B
=> estimated memory usage for worst case of 1e4 rays and 9 iterations = 1.03 GB <= FOR RAYS AND BUFFERS ONLY

**/

//Algorithm based on: Tomas Möller and Ben Trumbore, ``Fast, Minimum Storage Ray-Triangle Intersection'', journal of graphics tools, vol. 2, no. 1, pp. 21-28, 1997. 
int intersect_triangle(float3 O, float3 D,
		       float3 V0, float3 V1, float3 V2,
		       float *t, float *u, float *v)
{
	float3 E1, E2, T, P, Q;
	float DEN, iDEN;
	const float EPSILON_NUM = 0.000001; 
	
	// ray from orig in direction dir
	// ray line: R(t) = orig + t*dir
	// triangle eqn: T(u,v) = (1-u-v)*v0_ + u * v1_ + v * v2_
	// 		where u>=0, v>=0 and u+v<=1
	// solve for intersetion: R(t) = T(u,v)
	// [t,u,v] = 1/(P*E1) * [Q*E2,P*T,Q*D]
	// P = D x E2
	// Q = T x E1
	// E1 = V1 - V0
	// E2 = V2 - V0
	// T = O - V0
	
	
	//calculate triangle edge vectors. they define the triangle plane.
	E1 = V1 - V0;
	E2 = V2 - V0;

	P = cross(D, E2);
	DEN  = dot(P, E1);
	
	// DEN = 0 means that ray is parallel to triangle
	if (DEN > -EPSILON_NUM && DEN < EPSILON_NUM)
		return 0;
	
	iDEN = 1.0 / DEN;
	T    = O - V0;

	// REMEMBER: where u>=0, v>=0 and u+v<=1	
	*u = dot(P, T) * iDEN;
	if (*u < 0.0 || *u > 1.0)
		return 0;

	Q = cross(T, E1);

	// REMEMBER: where u>=0, v>=0 and u+v<=1	
	*v = dot(Q, D) * iDEN;
	if (*v < 0.0 || *u + *v > 1.0)
		return 0;

	// all conditions for u and v are met? now we can calculate t.
	*t = dot(Q, E2) * iDEN;

	return 1;
}

// postprocess intersect intermediate results. intersect code only colects closest intersect for every mesh. but which is the closest and is the ray entering or leaving it? where is the rays destination?
// those questions are answered here.
__kernel void intersect_postproc( __global const float3 *rays_origin, __global const float3 *rays_dir, __global float3 *rays_dest,
		__global int *rays_prev_isect_mesh_id, __global int *rays_n1_mesh_id, __global int *rays_n2_mesh_id,
		__global int *ray_entering, __global int *ray_isect_mesh_id, __global int *ray_isect_mesh_idx, __global const float3 *mesh_v0, 
		__global const float3 *mesh_v1, __global const float3 *mesh_v2, __global const int *mesh_id, __global const int *mesh_mat_type, __global float *isect_min_ray_len, 
		__global int *isects_count, __global int *ray_isect_mesh_idx_tmp, int mesh_count, int ray_count, float max_ray_len)                     
{
	//process results to calculate rays_dest and ray_entering
	const float EPSILON 	= 0.000001*max_ray_len; // EPSILON is used as a smallest possible length scale for error margins. for float32 this is ~7 digits.
					    // Therefore EPSILON should depnd on the current length scale of the scene. here max_ray_len is a good measure.
	int rid = get_global_id(0);	//ray index for parallel postprocessing
	float t_tmp = max_ray_len; 	//buffer for isect_min_ray_len
	int isect_mesh_id_tmp = -1;	//id of closest intersection
	int isect_mesh_idx_tmp = -1;	//index of closest intersected triangle
	float t_min = max_ray_len;	//intersection distance buffer
	int   n1_id = -1; //-1 means IOR_env
	int   n2_id = -1; //-1 means IOR_env
	
	int prev_isect = rays_prev_isect_mesh_id[rid]; // -2 means not initialized with material yet
			 			// -1 means located outside any mesh
			 			// >=0 means ray is in mesh with specific id given by the >=0 number

	//iterate over all meshes (not triangles) to find closest intersected mesh.	
	for(int j=0; j<mesh_count; j++){
		t_tmp=isect_min_ray_len[mesh_count*rid+j];//[j][rid];
		// update buffers if an intersection with a smaller distance is found
		if(t_tmp<t_min){
			t_min = t_tmp;
			isect_mesh_id_tmp = j;
			isect_mesh_idx_tmp = ray_isect_mesh_idx_tmp[mesh_count*rid+j];
		}
	}
		
	// if at least one intersect exists (at least one mesh id crossed) then figure out if the ray is entering or exiting the mesh.
	// an even number or intersects with a closed mesh (solid) means the ray is entering
	// an odd number means the opposite ;)
	int entering = 0;
	if(isect_mesh_id_tmp>=0){
		entering = 1-(isects_count[mesh_count*rid+isect_mesh_id_tmp] % 2);
		ray_entering[rid] = entering; 
		
		if(prev_isect == -2) //ray just started from source
		{	if(entering==1) //ray started in vacuum
			{	n1_id = -1;
				n2_id = isect_mesh_id_tmp;
			}
			else //n1 is obvouse but n2 still needs to be determined
			{	n1_id = isect_mesh_id_tmp;
				n2_id = -1; //needs to be determined later
			}
		
		}
		else if(prev_isect == -1)
		{	if(entering==1) //if entering a new mesh, n1 should be material of previous traversal
			{	n1_id = -1;
				n2_id = isect_mesh_id_tmp;
			}
			else //if exiting, THIS CASE IS IMPOSSIBLE FOR REFRACTIVE MEDIA. but these are still only IDs for materials
			{	n1_id = isect_mesh_id_tmp;
				n2_id = -1;
			}

		}
		else // previous intersect was some real material
		{	if(entering==1) //if entering a new mesh, n1 should be material of previous traversal
			{	n1_id = prev_isect;
				n2_id = isect_mesh_id_tmp;
			}
			else //if exiting
			{	n1_id = isect_mesh_id_tmp;
				n2_id = -1; //needs to be determined later
			}
		
		}
				
		// figure out which mesh id the ray is goint to propagate in next. we only need to do this if an intersection was found which is why we are in the if(isect_mesh_id_tmp>=0) case.
		// for this we need to separate 2 regions. 
		// 1. intersects that happen within the interval [t_min,t_min+EPSILON]. in this region only the final intersect counts.
		// 	LightPyCL currently does not consider interference effects. maybe in a later release.
		//	if the final intersect is entering, we have our n2 mesh.
		//	if the final intersect is exiting, we need to find the next mesh intersects outside of the defined interval which is handled by case 2.
		// 2. find the closest intersect within interval [t_min+EPSILON,max_ray_len] that is exiting. that mesh determines n2. if none exists, then n2=IOR_env.
		// NOTE: we must disregard all materials but refractive/dissipative for this to work. thus mesh_mat_type==0 must hold
		float t_minmin = t_min;
		float t_maxmin = t_min; //max ray len in interval [t_minmin,t_minmin+EPSILON]
		float t_minmax = max_ray_len; //ray length of closest exiting intersection in interval [t_min+EPSILON,max_ray_len]
		int   maxmin_entering = 0;
		int   maxmin_idx = -1;
		int   minmax_idx = -1;
		for(int j=0; j<mesh_count; j++){
			if(mesh_mat_type[j]==0 || mesh_mat_type[j]==4) //disregard intersections with any non-refractive media
			{	t_tmp=isect_min_ray_len[mesh_count*rid+j];
			
				//entering status of intersection j
				entering = 1-(isects_count[mesh_count*rid+j] % 2);

			
				//find largest intersection in [t_minmin,t_minmin+EPSILON]
				if(t_tmp<=t_minmin+EPSILON && t_tmp >= t_maxmin){
					t_maxmin = t_tmp;
					maxmin_idx = j;
					maxmin_entering = entering;
				}
			
				//find smallest exiting intersection in interal [t_min+EPSILON,max_ray_len]
				if(t_tmp>t_minmin+EPSILON && entering == 0 && t_tmp <= t_minmax){
					t_minmax = t_tmp;
					minmax_idx = j;
				}
			}
		}
		// now that we have knowledge of intersections in both intervals, we can check which case applies and determine n2.
		if(maxmin_entering == 1) //found adjacent mesh to enter
		{	t_min = t_maxmin;
			n2_id = maxmin_idx;			
		}
		else //exiting from one or many adjacent meshes
		{	if(maxmin_idx >= 0) //exiting one or many nearby meshes (first interval) means update of results is necesary
			{	t_min = t_maxmin;
			} //otherwise originally set values are ok => no need to do anything
		
			// we have exited one or many adjacent meshes. what meterial is the ray going to be in next? this does not alter t_min or intersect result buffers. only rays_n2_mesh_id.
			if(minmax_idx >= 0) //found mesh that has not yet been exited. this is the next material for ray to move into.
			{	n2_id = minmax_idx;
			}// ELSE: found no materials being exited in path. that means the ray is moving into the environment next => leave n2_id at its initial value of -1.
		}
		

	}

	//set results in STONE (global memory)
	rays_n1_mesh_id[rid] 	= n1_id;
	rays_n2_mesh_id[rid] 	= n2_id;
	rays_dest[rid] 		= rays_origin[rid] + rays_dir[rid] * t_min;
	ray_isect_mesh_id[rid] 	= isect_mesh_id_tmp;
	ray_isect_mesh_idx[rid] = isect_mesh_idx_tmp;
}

//intersect rays in parallel
__kernel void intersect( __global const float3 *rays_origin, __global const float3 *rays_dir, __global float3 *rays_dest, __global int *ray_entering, __global int *ray_isect_mesh_id, __global int *ray_isect_mesh_idx, __global const float3 *mesh_v0, __global const float3 *mesh_v1, __global const float3 *mesh_v2, __global const int *mesh_id, __global float *isect_min_ray_len, __global int *isects_count, __global int *ray_isect_mesh_idx_tmp, int mesh_count, int tri_count, int ray_count, float max_ray_len)                     
{                                                                            	
	const float EPSILON 	= 0.000001*max_ray_len; // EPSILON is used as a smallest possible length scale for error margins. for float32 this is ~7 digits.
					    // Therefore EPSILON should depnd on the current length scale of the scene. here max_ray_len is a good measure.
	int rid 		= get_global_id(0); //ray index
	float3 ray_origin 	= rays_origin[rid];
	float3 ray_dir 		= rays_dir[rid];
	int isects_count_tmp 	= 0;
	float isect_min_ray_len_tmp = max_ray_len;
	int ray_isect_mesh_idx_ltmp = -1;
	float t,u,v;	
	int isect 		= 0;
	int idx_tmp 		= 0;
	
	//iterate this ray over all triangles and find intersections. sorry, I can not think of any speedup here but brute force. 
	//rays can originate everywhere and point anywhere and we need to know all intersections. imho this problem can only be bruteforced.
	for(int i = 0; i < tri_count; i++){
		idx_tmp = mesh_count*rid+mesh_id[i];		
		// detect if we are moving to the next mesh (solid). if so, move mesh results to global memory and start fresh temp buffers for new mesh.
		if(i>0 && mesh_id[i-1] != mesh_id[i]){
			isects_count[idx_tmp-1] 		= isects_count_tmp;
			isect_min_ray_len[idx_tmp-1] 		= isect_min_ray_len_tmp;
			ray_isect_mesh_idx_tmp[idx_tmp-1]	= ray_isect_mesh_idx_ltmp;
			isects_count_tmp 			= 0;
			isect_min_ray_len_tmp 			= max_ray_len;
			ray_isect_mesh_idx_ltmp 		= -1;
		}
		
		//intersect ray with triangle
		isect = intersect_triangle(ray_origin, ray_dir, 
					mesh_v0[i], mesh_v1[i], mesh_v2[i], 
					&t, &u, &v);
					
		//if intersect is found check if it is the one with shortest distance to ray origin and update isect_min_ray_len and isects_count of according mesh_id.
		if(isect && t > EPSILON){ 
			if(t < isect_min_ray_len_tmp){
				isect_min_ray_len_tmp = t;
				ray_isect_mesh_idx_ltmp = i;
			}
			isects_count_tmp += 1;
		}
	}
	//make sure results for the last mesh end up in global mem too!
	isects_count[idx_tmp] 		= isects_count_tmp;
	isect_min_ray_len[idx_tmp] 	= isect_min_ray_len_tmp;
	ray_isect_mesh_idx_tmp[idx_tmp] = ray_isect_mesh_idx_ltmp;	
}


//reflect and refract according to snells equations for unpolarized light. no need for polarized light yet. if pol dependence is required the tracer needs an overhawl.
int reflect_refract(float3 in_ray_dest, float3 in_ray_dir, float in_ray_power, 
	float3 *ray_reflect_origin, float3 *ray_reflect_dir, float *ray_reflect_power, int *ray_reflect_measured,
	float3 *ray_refract_origin, float3 *ray_refract_dir, float *ray_refract_power, int *ray_refract_measured,
	float3 surf_normal_in, float n1, float n2) 
{
    float3 surf_normal = surf_normal_in;
    float r = n1/n2;
    float TIR_check;
    
    //sanitize the surface normal vector. surf vector must always be pointing towards the ray to give correct results.
    float cosT1 = -dot(surf_normal,in_ray_dir);
    if(cosT1 < 0.0){
    	surf_normal = -surf_normal_in;
    	cosT1 = -dot(surf_normal,in_ray_dir);
    }

    TIR_check = 1.0 - pown(r,2) * (1.0 - pown(cosT1,2));
    if (TIR_check >= 0.0) //check for total internal reflection
	{	// normal refract
	    	float cosT2 = sqrt(TIR_check);
	    	float Rs = pown(fabs((n1*cosT1 - n2*cosT2)/(n1*cosT1 + n2*cosT2)),2); //s polarized
	    	float Rp = pown(fabs((n1*cosT2 - n2*cosT1)/(n1*cosT2 + n2*cosT1)),2); //p polarized
	    	
	    	float reflect_power = in_ray_power * (Rs+Rp)/2.0;
	    	
	    	*ray_reflect_dir      = in_ray_dir+2.0*cosT1*surf_normal;
    		*ray_reflect_origin   = in_ray_dest;
	    	*ray_reflect_power    = reflect_power;
	    	*ray_reflect_measured = 0;
	    	
		*ray_refract_dir      = in_ray_dir * r + (r * cosT1 - cosT2) * surf_normal;
		*ray_refract_origin   = in_ray_dest;
		*ray_refract_power    = in_ray_power-reflect_power;
		*ray_refract_measured = 0;
		return 0;
	}
	if(TIR_check < 0.0) // total internal reflection occured => propagate all power to reflection.
	{
		*ray_reflect_dir      = in_ray_dir+2.0*cosT1*surf_normal;
    		*ray_reflect_origin   = in_ray_dest;
	    	*ray_reflect_power    = in_ray_power;
	    	*ray_reflect_measured = 0;
	    	
		*ray_refract_dir      = (float3)(0,0,0);;
		*ray_refract_origin   = in_ray_dest;
		*ray_refract_power    = 0.0;
		*ray_refract_measured = -1;
		return 0;
	}
	return 1;
}

// reflect and refract rays while checking if the ray should be terminated or measured and sets result buffers accordingly.
__kernel void reflect_refract_rays( __global const float3 *in_rays_origin, __global const float3 *in_rays_dest, __global const float3 *in_rays_dir, 
		__global float *in_rays_power, __global int *in_rays_measured, __global const int *in_ray_entering,
		__global int *rays_n1_mesh_id, __global int *rays_n2_mesh_id,
		__global float3 *rays_reflect_origin, __global float3 *rays_reflect_dir, 
		__global float *rays_reflect_power, __global int *rays_reflect_measured, 
		__global float3 *rays_refract_origin, __global float3 *rays_refract_dir, 
		__global float *rays_refract_power, __global int *rays_refract_measured, 
		__global int *ray_isect_mesh_id, __global int *ray_isect_mesh_idx, 
		__global const float3 *mesh_v0, __global const float3 *mesh_v1, __global const float3 *mesh_v2, 
		__global const int *mesh_id, __global const int *mesh_mat_type, __global const float *mesh_ior,
		__global const float *mesh_refl, __global const float *mesh_diss, float IOR_env, int mesh_count, int ray_count, float max_ray_len)                     
{	const float EPSILON 	= 0.000001*max_ray_len; // EPSILON is used as a smallest possible length scale for length error margins. for float32 this is ~7 digits.
					    // Therefore EPSILON should depnd on the current length scale of the scene. here max_ray_len is a good measure.
	const float EPSILON_NUM	= 0.000001; // EPSILON_NUM not for length comparisons can be much smaller because here values close to 0 are of interest.
	int rid = get_global_id(0);
	int rmid = ray_isect_mesh_id[rid];
	
	//Default values to terminate ray. because nothing was hit.
	int   mesh_mat = 2; // type of mesh material. 0 refractive, 1 mirror, 2 terminate, 3 measure and 4 anisotropic refractive
	//float IOR_mesh = 1.0; 	
	float R_mesh   = 0.0;
	//float D_mesh   = 0.0;
	
	// if ray was intersected then get values from mesh material parameters
	if(rmid >= 0)
	{	mesh_mat = mesh_mat_type[rmid];
		//IOR_mesh = mesh_ior[rmid]; 	
		R_mesh   = mesh_refl[rmid];
		//D_mesh   = mesh_diss[rmid];
	}

	// calculate dissipation and determine IOR of material the ray is impinging from
	int r_orig_mid = rays_n1_mesh_id[rid]; //n1 mesh id
	int r_dest_mid = rays_n2_mesh_id[rid]; //n2 mesh id
	float IOR_in_ray = IOR_env;
	float IOR_n2_ray = IOR_env;
	float in_ray_pow = in_rays_power[rid];
	float3 in_ray_dest = in_rays_dest[rid];
	
	if( r_orig_mid >= 0)
	{	IOR_in_ray = mesh_ior[r_orig_mid];
		
		//if material in which input ray was dissipative then modify inray power
		if(mesh_mat_type[r_orig_mid] == 0 && mesh_diss[r_orig_mid] > EPSILON_NUM)
		{	float ray_len = length(in_ray_dest - in_rays_origin[rid]);
			in_ray_pow = in_ray_pow * exp(-mesh_diss[r_orig_mid]*ray_len);
			in_rays_power[rid] = in_ray_pow;
		}
	}
	else
	{	IOR_in_ray = IOR_env;
		
	}	
	
	// if ray destination material is not air (id=-1) then fetch IOR value.
	if(r_dest_mid >= 0)
	{	IOR_n2_ray = mesh_ior[r_dest_mid];	
	}

	//if ray_isect_mesh_idx == -1 then no intersects exist and ray can also be terminated
	//if ray has not been terminated by measurement or termination sruface => generate reflect and refract beams.
	int irm = in_rays_measured[rid]; 
	if(irm==0 && rmid >= 0 && (mesh_mat == 0 || mesh_mat == 1)){			
		int m_idx = ray_isect_mesh_idx[rid];
		float3 v0 = mesh_v0[m_idx];
		float3 v1 = mesh_v1[m_idx];
		float3 v2 = mesh_v2[m_idx];
		float3 surf_normal = normalize(cross(v1-v0,v2-v1));
			
		float3 r_reflect_origin;
		float3 r_reflect_dir;
		float r_reflect_power;
		int r_reflect_measured;
		float3 r_refract_origin;
		float3 r_refract_dir;
		float r_refract_power;
		int r_refract_measured;
		
		reflect_refract(in_ray_dest, in_rays_dir[rid], in_ray_pow, 
			&r_reflect_origin, &r_reflect_dir, &r_reflect_power, &r_reflect_measured,
			&r_refract_origin, &r_refract_dir, &r_refract_power, &r_refract_measured,
			surf_normal, IOR_in_ray, IOR_n2_ray);
		
		if(mesh_mat == 0) // not a mirror
		{	rays_reflect_origin[rid]	= r_reflect_origin;
			rays_reflect_dir[rid]		= r_reflect_dir;
			rays_reflect_power[rid]		= r_reflect_power;
			rays_reflect_measured[rid]	= r_reflect_measured;
			
			rays_refract_origin[rid]	= r_refract_origin;
			rays_refract_dir[rid]		= r_refract_dir;
			rays_refract_power[rid]		= r_refract_power;
			rays_refract_measured[rid]	= r_refract_measured;
		}
		else // mirror material
		{	rays_reflect_origin[rid]	= r_reflect_origin;
			rays_reflect_dir[rid]		= r_reflect_dir;
			rays_reflect_power[rid]		= in_ray_pow * R_mesh; // Mirror Losses
			rays_reflect_measured[rid]	= r_reflect_measured;
			
			rays_refract_origin[rid]	= r_refract_origin;
			rays_refract_dir[rid]		= (float3)(0,0,0);
			rays_refract_power[rid]		= 0.0;
			rays_refract_measured[rid]	= -1;

		
		}
		
	}
	else{ //if rays go nowhere, were measured or terminated the reflected and refracted rays need not be calculated but need termination.
		rays_reflect_origin[rid]	= in_rays_dest[rid];
		rays_reflect_dir[rid]		= (float3)(0,0,0);
		rays_reflect_power[rid]		= 0.0;
		rays_reflect_measured[rid]	= -1;
	
		rays_refract_origin[rid]	= in_rays_dest[rid];
		rays_refract_dir[rid]		= (float3)(0,0,0);
		rays_refract_power[rid]		= 0.0;
		rays_refract_measured[rid]	= -1;

		if(mesh_mat == 2 || rmid < 0){ //beam hit termination surface or goes nowhere
			in_rays_measured[rid] = -1;
		}
		if(mesh_mat == 3 && rmid >=0 ){ // beam hit measurement surface
			in_rays_measured[rid] = 1;
		}
	}
	
}

// rotate an array of vectors
__kernel void rotate_vec( __global const float3 *vecs, __global const float3 *rot_mtx, 
		__global float3 *pivot, __global float3 *vecs_rot)  
{
	int gid = get_global_id(0);
	float3 piv = pivot[0];
	float3 vec = vecs[gid] - piv;
	vecs_rot[gid] = (float3)(dot(vec, rot_mtx[0]),dot(vec, rot_mtx[1]),dot(vec, rot_mtx[2]) ) + piv;
}


// perform stereographic projection of input vectos located on a hemesphere surface
__kernel void stereograph_project( __global const float3 *vecs, __global const float *pwrs, __global const float3 *rot_mtx, __global float3 *pivot, __global float *x, __global float *y, __global float *pwrs_cor)  
{
	int gid = get_global_id(0);
	float3 piv = pivot[0];
	float3 vec = vecs[gid] - piv;
	float  pwr = pwrs[gid];
	float u = dot(vec, rot_mtx[0]) + piv.x;
	float v = dot(vec, rot_mtx[1]) + piv.y;
	float w = dot(vec, rot_mtx[2]) + piv.z;
	float l = sqrt(pown(u,2)+pown(v,2)+pown(w,2));
	
	float xt = u/(l+w);
	float yt = v/(l+w);
	float A  = 4.0 / pown((1.0 + pown(xt,2) + pown(yt,2)),2);
	
	x[gid] = xt;
	y[gid] = yt;
	pwrs_cor[gid]  = pwr/A;
} 

// perform an azimuth/elevation mapping of vecs located on sphere surface to a circle and compensate surface element for acurate power calculation.
__kernel void angular_project( __global const float3 *vecs, __global const float *pwrs, __global const float3 *rot_mtx, __global float3 *pivot, __global float *x, __global float *y, __global float *pwrs_cor)  
{	const float EPSILON 	= 0.000001;
	int gid = get_global_id(0);
	float3 piv = pivot[0];
	float3 vec = vecs[gid] - piv;
	float  pwr = pwrs[gid];
	float u = dot(vec, rot_mtx[0]) + piv.x;
	float v = dot(vec, rot_mtx[1]) + piv.y;
	float w = dot(vec, rot_mtx[2]) + piv.z;
	float l = sqrt(pown(u,2)+pown(v,2)+pown(w,2));
	float3 vecr = (float3)(u,v,w)/l;
	float cosT = dot((float3)(0,0,1),vecr);
	float phi  = atan2(v,u);
	
	float r = acos(cosT);
	float xt = r * cos(phi);
	float yt = r * sin(phi);
	
	// dA_Tp = sinT * dT * dphi -> dA_rp = r * dr * dphi = T * dT * dphi = dx*dy
	// => dA_Tp = sinT / T * dx*dy => A=sinT/T
	float A  = 1.0;
	if(r > EPSILON) //prevent calculation errors of sin(x)/x for x~0
	{	A  = sin(r)/r;
	}

	
	x[gid] = xt;
	y[gid] = yt;
	pwrs_cor[gid]  = pwr/A;
} 

LightPyCL
=========

<img src="https://github.com/lillg/LightPyCL/blob/master/logo2.png"
 alt="LightPyCL Logo" title="LightPyCL" align="right" />

LightPyCL is a high-performance physical 3D raytracer written in Python/PyOpenCL for the evaluation of optical elements with arbitrary 3D geometry. LightPyCL is intended to simulate the behaviour of intricate optical elements and systems illuminated by light sources with arbitrary directional characteristics and configurations. LightPyCL is not intended for rendering pretty graphics but could potentially be used to create accurate illumination maps in a timely manner. 

LightPyCL is released under the GPLv3 and is free software released in hope that it will be useful to someone.

## Features

* Simplicity:
	* Python is used for scene scripting and data pre- and post-processing.
	* With a few lines of Python code a scene can be set up and simulated.
	* With a few more lines the results can be processed and displayed.
* Performance: 
	* Calculations are done with OpenCL.
	* Calculations can be efficiently run on CPUs or GPUs.
	* 1,000,000 rays to accurately sample a scene. Not a problem!
	* 100,000 polygons on top of that. Easy! 
* Results can be stored in a Python pickle for further evaluation at a later time.
* DXF output of traced and untraced scenes.
* LightPyCL supports positive index materials, mirrors, termination surfaces and measurement surfaces.
* Any material can be applied to any mesh. This means you can 
	* measure the spatial distribution of light on any kind of surface geometry.
	* terminate rays anywhere in the scene on an arbitrary shape.
	* reflect and refract light from/into an arbitrarily shaped object.
* Unpolarized rays are propagated in a physically correct fashion.
* Power transmission and directivity analysis for unpolarized light.
* Basic mesh transformations on optical elements.
* Leveraging of scene symmetries to simulate many light sources from results of one.
* Generate optical elements directly from Python either by revolving a 2D curve or parametrically generating a mesh.

## Prerequisites

To get LightPyCl to work, you will need the following:

1. [PyOpenCL](http://mathema.tician.de/software/pyopencl/) for the heavy lifting.
Installing instructions for PyOpenCL can be found [here](http://wiki.tiker.net/PyOpenCL/Installation/Linux).
2. [dxfwrite](https://pypi.python.org/pypi/dxfwrite/) for 3D geometry output.
3. python-numpy and python-matplotlib.
4. Some DXF viewer (like [g3dviewer](http://automagically.de/g3dviewer/) or [gCAD3D](http://www.gcad3d.org/)) to view saved geometry and traced scenes.


## Usage

To get LightPyCL to run, you need to set up your light sources and optical elements. Additionally you can set up surfaces that terminate rays that don't need to be traced further or surfaces that terminate rays and measure their intensity simultaneously in order to measure directional characteristics.

Some examples are included in the repository to help you get started.

### Light sources

To create a light source you will need to import the *light_source* class

`import light_source`

and create a light source

```
ray_count = 10000
directivity = lambda azimuth,elevation: np.cos(elevation)

lightsource0 = light_source.light_source(center=[-1,0,0,0], direction=[1,0,1], directivity=directivity, power=1.0, ray_count=ray_count)
lightsource1 = light_source.light_source(center=[1,0,0,0], direction=[-1,0,1], directivity=directivity, power=2.0, ray_count=ray_count)

lightsources = [lightsource0, lightsource1]
```

*ray_count* defines the number of rays the light source is to emit, and *directivity* is the directional power distribution function of the light source. Note that elevation is measured away from the *direction* the light source is pointing in.

The first light source origin is placed on the left hand side of the scene (x=-1, y and z = 0) and the second source is on the right (x=+1).

The first source is pointing to the upper right [1,0,1] and the second source is pointing up left [-1,0,1] with a respective power of 1.0 and 2.0 and the same number of rays.

Finally, all the light sources are added to the list *lightsources* which can be handed to the raytracer.

### Optical elements and scene objects

To generate optical elements, an instance of *geo_optical_elements* is required

```
import geo_optical_elements as goe
oe = goe.optical_elements()
```

With *oe* we can currently generate primitives such as cubes, spheres, hemispheres and cylinders as well as more complex meshes such as parabolic mirrors, spherical lenses and other surfaces of revolution from 2D curves.

To create a hemisphere intended to measure a directional pattern of our light sources, *oe.hemisphere* is called
 
```
hemisphere = oe.hemisphere(center=[0,0,0,0],radius=500.0,IOR=0.0)
hemisphere.setMaterial(mat_type="measure")
```

The *center* of the hemisphere is placed at [0,0,0] and its *radius* is 500. To define it as a measuring surface, the *setMaterial* method is called with *mat_type="measure"*. A termination surface would *mat_type="terminator"*. 

Typical materials would be of *mat_type="refractive"* with a floating point *IOR* and *dissipation* or of *mat_type="mirror"* with a *reflectivity* value and would be set up like this

```
cube = oe.cube(center=(0,0,-0.01,0),size=[100,100,10,0])
cube.setMaterial(mat_type="refractive",IOR=1.5, dissipation=0.01)
cube.setMaterial(mat_type="mirror",reflectivity=0.98)
```

Note that dissipation is to be given in the base length unit you decide to use in the simulation. In LightPyCL length has no unit, therefore you can decide to make a length of 1 to be 1 cm. In the case that a length of 1 is 1 cm you would have to give your dissipation in 1/cm.

To create a parabolic mirror with a focal point *focus* at [0,0,0] and a *focal_length* of 5 and a dish *diameter* of 20, *oe.parabolic_mirror* can be used as follows

`parabolic_mirror = oe.parabolic_mirror(focus=[0,0,0],focal_length=5.0,diameter=20.0,reflectivity = 0.98)`

Optical elements can be moved and rotated as follows:

```
parabolic_mirror.rotate(axis="y",angle=-np.pi/2,pivot = (0,0,0,0))
parabolic_mirror.translate([1,1,0])
```

Note that *translate* performs only a relative displacement of the mesh.

Finally, after generating and positioning the optical elements, they must be assembled into a *scene* for the raytracer

`scene = [hemisphere, parabolic_mirror, cube]`

The raytracer then converts all the meshes into a geometry buffer that is sent to the CL device.

### Initializing and running the raytracer

The raytracer is an iterative tracer that takes a set of rays as an input and intersects them with the scene geometry. The closest intersections are used as the origin for reflected and refracted rays which are the output of one iteration. The output rays of an iteration (intersection and reflection/refraction cycle) are pruned by removing all measured and terminated rays and then used as the new input rays for the next iteration. Before the ray pruning the intersected input rays and the reflected and refracted rays are collected in a result buffer for later processing.

To use the raytracer the class *iterative_tracer* must be imported and initialized

```
import numpy as np
import iterative_tracer as it
tracer = it.CL_Tracer(platform_name="NVIDIA",device_name="770")
```

This tells the *tracer* to use an nVidia GTX770 card. Alternatively, if the tracer should be run on a CPU the *platform_name*="AMD" and *device_name*="i5" can be selected. If you provide an empty or invalid *platform_name* and *device_name* string the first device of the first platform is used. Note that the *platform_name* and *device_name* string must only be contained within the full platform and device specifiers.

Once initialized, the *tracer* can be run with the following command

```
iterations  = 8
max_ray_len = 1e3
ior_env     = 1.0
tracer.iterative_tracer(light_source=lightsources,meshes=scene,trace_iterations=iterations,max_ray_len=max_ray_len,ior_env=ior_env)
```

*iterations* tells the raytracer to follow the rays to a depth of 8.
*max_ray_len = 1e3* specifies the maximum length a ray can have if it does not intersect.
*ior_env     = 1.0* determines the refractive index of the environment (1.0 for vacuum).

Once the tracer has completed its task, the results can be found in *tracer.results*. The resulting traced scene can be saved to a DXF file with *tracer.save_traced_scene("./results.dxf")*. Note that saving the file can take a long time when large amounts of rays have been traced.

To access only the final position and intensity of the measured rays you can call *tracer.get_measured_rays()* which will return a tuple *(pos,pow)* of the vertex array *pos* and float array *pow*.

### Postprocessing and plotting

When the tracing is done, the results can be analysed in a 2D directivity plot with stereographic mapping or with linear elevation and azimuth mapped to a circle surface. The measured data can also be plotted to a 1D azimuth/elevation histogram. All plots are corrected for mapping distortion and should give the correct power values for their corresponding surface and line elements.

With 

```
nf=2.0
m_surf_extent=((-np.pi/nf,np.pi/nf),(-np.pi/nf,np.pi/nf))
m_points=100 
tracer.plot_binned_data(limits=m_surf_extent,points=m_points,use_angular=True,use_3d=True)
```

you will get a 3D surface plot of the power distribution along the measurement surface mapped to elevation and azimuth of the ray measured from [0,0,0] in the scene and an elevation extent of +-pi/2.

With 

`tracer.plot_elevation_histogram(points=90,pole=[0,0,1,0])`

you can show an aggregated elevation plot that assumes rotational symmetry around the axis defined by *pole* and 90 bins.

If you need to know the beam width of your light source or your light source passing through some optics, you can use 

`tracer.get_beam_width_half_power(points=90,pole=[0,0,1,0])`

to find out at the angle (elevation) of the cone into which half the beams power is emitted.

Alternatively, you can use

`tracer.get_beam_HWHM(points=90,pole=[0,0,1,0])`

to find the beams half width at half maximum. Note however that *get_beam_HWHM*, depending on the scene, requires a large number of rays to deliver accurate results.

If you need any other evaluative tools that are not included currently, feel free to write up a few lines of code. The results are located in *tracer.results* as a list of the tuple *(ray_origins,ray_destinations,ray_powers,state_measured)*. *ray_origins[i]*, *ray_destinations[i]*, *ray_powers[i]* and *state_measured[i]* represent one ray and *state_measured[i]* indicates if the ray was terminated (*state_measured[i]=-1*), measured (*state_measured[i]=+1*) or still traversing the scene and thus neither measured or terminated (*state_measured[i]=0*).

Please feel free to contribute your evaluation code or any other code/improvement to LightPyCL for that matter to the LightPyCL project.

### Putting it all together

The files *example_directivity_lens.py*, *example_directivity_parabolic_mirror.py* and *example_directivity_sphericalMeasSurf.py* provided with LightPyCL are constructed just as described in the previous subsections of this *README* and are ready to run. To try out the tracer examples just run

`python example_directivity_lens.py`,

`python example_directivity_parabolic_mirror.py` 

or

`python example_directivity_sphericalMeasSurf.py` 

in your favourite terminal emulator.


## Performance

The performance of the raytracer is determined by measuring the time __T__ a combined intersection and reflection/refraction cycle takes for __N__ input rays and __M__ triangles in a scene. Because every ray has to search all triangles in a scene for a valid closest intersection, __N__ * __M__ gives the number of performed refractive intersections or, put differently, the number of rays that could be intersected and refracted if the scene consisted of one triangle. Thus a comparative measure of performance is __N__ * __M__ / __T__ given in "refractive intersections/s" or "RI/s".

Here are some results from various platforms:
<table>
	<tr><td>Intel i5</td>		<td>~ 0.5e9 RI/s</td></tr>
	<tr><td>nVidia GTX460</td>	<td>~ 4.1e9 RI/s</td></tr>
	<tr><td>nVidia GTX770</td>	<td>~ 9.9e9 RI/s</td></tr>
</table>

Performance results are printed in the console during simulation, so if you would like to share those results, drop me a line!


LightPyCL
=========

LightPyCL is a high performance physical 3D ray-tracer written in Python/PyOpenCL for the evaluation of optical elements with arbitrary 3D geometry. LightPyCL is intended to simulate the behaviour of intricate optical elements and systems when illuminated by light sources with arbitrary directional characteristics and configurations. LightPyCL is not intended for rendering pretty graphics but could potentially be used to create accurate illumination maps in a timely manner. 

LightPyCL is released under the GPLv3 and is free software released in hope that it will be useful to someone.

## Features

* Simplicity:
	* python is used for scene scripting and data pre- and post-processing.
	* with a few lines of python code a scene can be setup and simulated.
	* with a few more lines the results can be processed and displayed.
* Performance: calculations are done with OpenCL 
	* calculations can be efficiently run on CPUs or GPUs.
	* 1 000 000 rays to accurately sample a scene. Not a problem!
	* 100 000 polygons on top of that. Easy! 
* Results can be stored in a python pickle for further evaluation at a later time.
* DXF output of traced and untraced scenes.
* LightPyCL supports positive index materials, mirrors, termination surfaces and measurement surfaces.
* Any material can be applied to any mesh. This means you can 
	* measure the spatial distribution of light on any kind of surface geometry.
	* terminate rays anywhere in the scene on an arbitrary shape.
	* reflect and refract light from/into an arbitrarily shaped object.
* Unpolarized rays are propagated in a physically correct fashion.
* Power transmission and directivity analysis for unpolarized light.
* Basic mesh transformations on optical elements.
* Leverageing of scene symmetries to simulate many light sources from results of one.
* Generate optical elements directly from python either by revolving a 2D curve or parametrically generating a mesh.

## Prerequisites

To get LightPyCl to work, you will need the following things.

1. [PyOpenCL](http://mathema.tician.de/software/pyopencl/) for the heavy lifting.
Installing instructions for PyOpenCL can be found [here](http://wiki.tiker.net/PyOpenCL/Installation/Linux).
2. [dxfwrite](https://pypi.python.org/pypi/dxfwrite/) for 3D geometry output.
3. python-numpy and python-matplotlib.
4. Some DXF viewer (like [g3dviewer](http://automagically.de/g3dviewer/) or [gCAD3D](http://www.gcad3d.org/)) to view saved geometry and traced scenes.


## Usage

To get LightPyCL to run you need to setup your light sources and optical elements. Additionally you can setup surfaces that terminate rays that don't need to be traced further or surfaces that terminate rays and measure their intensity simultaneously in order to measure directional characteristics.

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

*ray_count* defines the amount of rays the light source is to emit and *directivity* is the directional power distribution function of the light source. Note that elevation is measured away from the *direction* the light source is pointing in.

The first light source origin is placed on the left hand side of the scene (x=-1, y and z = 0) and the second source is on the right (x=+1).

The first source is pointing to the upper right [1,0,1] and the second source is pointing up left [-1,0,1] with a respective power of 1.0 and 2.0 and the same amount of rays.

Finally, all the light sources are added to the list *lightsources* which can be handed to the raytracer.

### Optical elements and scene objects

To generate optical elements the an instance of *geo_optical_elements* is required

```
import geo_optical_elements as goe
oe = goe.optical_elements()
```

With *oe* we can currently generate primitives such as cubes, spheres, hemispheres and cylinders as well as more complex meshes such as parabolic mirrors, spherical lenses and other surfaces of revolution from 2D curves.

To create a hemisphere intended to measure a directional pattern of our light sources *oe.hemisphere* is called
 
`hemisphere = oe.hemisphere(center=[0,0,0,0],radius=500.0,IOR=0.0)`

The *center* of the hemisphere is placed at [0,0,0] and its *radius* is 500. To define it as a measuring surface it is assigned an *IOR* (index of refraction) of 0.0. A termination surface would have an *IOR* of -1.0 and typical materials would be assigned an *IOR* > 1.0. Note that currently *IOR* values can not be greater than 1000.

To create a parabolic mirror with a focal point *focus* at [0,0,0] and a *focal_length* of 5 and a dish *diameter* of 20 *oe.parabolic_mirror* can be used as follows

`parabolic_mirror = oe.parabolic_mirror(focus=[0,0,0],focal_length=5.0,diameter=20.0,reflectivity = 0.98)`

Note that the *reflectivity* is encoded in the *IOR* value of the *GeoObject* in order to keep the raytracer code simple. This is also the reason why *IOR* > 1000 is not permitted for refractive materials as an *IOR* = 1000.98 would mean "Mirror with reflectivity of 98%" to the raytracer.

Optical elements can be moved and rotated as follows:

```
parabolic_mirror.rotate(axis="y",angle=-np.pi/2,pivot = (0,0,0,0))
parabolic_mirror.translate([1,1,0])
```

Note that *translate* performs only a relative displacement of the mesh.

Finally, after generating and positioning the optical elements, they must be assembled into a *scene* for the raytracer

`scene = [hemisphere, parabolic_mirror]`

The raytracer then converts all the meshes into a geometry buffer that is sent to the CL device.

### Initializing and running the raytracer

The raytracer is an iterative tracer that takes a set of rays as an input and intersects them with the scene geometry. The closest intersections are used as the origin for reflected and refracted rays which are the output of one iteration. The output rays of an iteration (intersection and reflection/refraction cycle) are pruned by removing all measured and terminated rays and then used as the new input rays for the next iteration. Before the ray pruning the intersected input rays and the reflected and refracted rays are collected in a result buffer for later processing.

To use the raytracer the class *iterative_tracer* must be imported and initialized

```
import numpy as np
import iterative_tracer as it
tracer = it.CL_Tracer(platform_name="NVIDIA",device_name="770")
```

this tells the *tracer* to use an nVidia GTX770 card. Alternatively, if the tracer should be run on a CPU the *platform_name*="AMD" and *device_name*="i5" can be selected. If you provide an empty or invalid *platform_name* and *device_name* string the first device of the first platform is used. Note that the *platform_name* and *device_name* string must only be contained within the full platform and device specifiers.

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

### Putting it all together

The files *example_directivity_lens.py*, *example_directivity_parabolic_mirror.py* and *example_directivity_sphericalMeasSurf.py* provided with LightPyCL are constructed just as described in the previous subsections of this *README* and are ready to run. To try out the tracer examples just run

`python example_directivity_lens.py`,

`python example_directivity_parabolic_mirror.py` 

or

`python example_directivity_sphericalMeasSurf.py` 

in your favourite terminal emulator.


## Performance

The performance of the raytracer is determined by measuring the time __T__ a combined intersection and reflection/refraction cycle takes for __N__ input rays and __M__ triangles in a scene. Because every ray has to search all triangles in a scene for a valid closest intersection, __N__ * __M__ gives the amount of performed refractive intersections or put differently the amount of rays that could be intersected and refracted if the scene consisted of one triangle. Thus a comparative measure of performance is __N__ * __M__ / __T__ given in "refractive intersections/s" or "RI/s".

Here are some results from various platforms:
<table>
	<tr><td>Intel i5</td>		<td>~ 0.5e9 RI/s</td></tr>
	<tr><td>nVidia GTX460</td>	<td>~ 4.1e9 RI/s</td></tr>
	<tr><td>nVidia GTX770</td>	<td>~ 9.9e9 RI/s</td></tr>
</table>

Performance results are printed in the console during simulation, so if you would like to share those results, drop me a line!


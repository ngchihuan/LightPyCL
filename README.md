LightPyCL
=========

LightPyCL is a high performance physical 3D ray-tracer written in Python/PyOpenCL for the evaluation of optical elements with arbitrary 3D geometry.

## Features

* Simplicity:  python is used for scene scripting and data pre- and post-processing.
* Performance: calculations are done with OpenCL 
	* calculations can be efficiently run on CPUs or GPUs.
* Results can be stored in a python pickle for further evaluation at a later time.
* Traced and untraced scenes can be saved as DXF files.
* LightPyCL positive index materials, mirrors, termination surfaces and measurement surfaces.
* Any material can be applied to any mesh. This means you can 
	* measure the spatial distribution of light on any kind of surface geometry.
	* terminate rays anywhere in the scene on an arbitrary shape.
	* reflect and refract light from/into an arbitrarily shaped object.
* Unpolarized rays are propagated in a physically correct fashion (reflection/refraction/total internal reflection)
* Power transmission and directivity analysis for unpolarized light.
* Basic mesh transformations can be performed on optical elements.
* Simulate many light sources efficiently by using a single light source and scene symmetries.
* Generate optical elements directly from python either by revolving a 2D curve or parametrically generating a mesh.

## Prerequisites

To get LightPyCl to work, you will need the following things.

1. [PyOpenCL](http://mathema.tician.de/software/pyopencl/) for the heavy lifting.
Installing instructions for PyOpenCL can be found [here](http://wiki.tiker.net/PyOpenCL/Installation/Linux).

2. [dxfwrite](https://pypi.python.org/pypi/dxfwrite/) for 3D geometry output.

3. python-numpy and python-matplotlib.

4. Some DXF viewer (like [g3dviewer](http://automagically.de/g3dviewer/) or [gCAD3D](http://www.gcad3d.org/)).


## Usage

To get LightPyCL to run you need to setup your light sources and optical elements. Additionally you can setup surfaces that terminate rays or surfaces that terminate rays and measure their intensity.

Some examples are included in the repo to help you get started.

### Light sources

To create a light source you will need to import the light_source class

'import light_source'

and create a light source

'''
ray_count = 10000
directivity = lambda azimuth,elevation: np.cos(elevation)

lightsource0 = light_source.light_source(center=[-1,0,0,0], direction=[1,0,1], directivity=directivity, power=1.0, ray_count=ray_count)
lightsource1 = light_source.light_source(center=[1,0,0,0], direction=[-1,0,1], directivity=directivity, power=2.0, ray_count=ray_count)

lightsources = [lightsource0, lightsource1]
'''

ray_count defines the amount of rays the light source is to emit and directivity is the directional power distribution function of the light source. Note that elevation is measured away from the direction the light source is pointing in.

The first light source origin is placed on the left hand side of the scene (x=-1, y and z = 0) and the second source is on the right (x=+1).

The first source is pointing to the upper right [1,0,1] and the second source is pointing up left [-1,0,1] with a respective power of 1.0 and 2.0 and the same amount of rays.

Finally, all the light sources are added to the list 'lightsources' with can be handed to the raytracer.

### Optical elements and scene objects

To generate optical elements the an instance of 'geo_optical_elements' is required

'''
import geo_optical_elements as goe
oe = goe.optical_elements()
'''

With 'oe' we can currently generate primitives such as cubes, spheres, hemispheres and cylinders as well as more complex meshes such as parabolic mirrors, spherical lenses and other surfaces of revolution from 2D curves.

To create a hemisphere intended to measure a directional pattern of our light sources oe.hemisphere is called
 
'hemisphere = oe.hemisphere(center=[0,0,0,0],radius=500.0,IOR=0.0)'

The center of the hemisphere is placed at [0,0,0] and its radius is 500. To define it as a measuring surface it is assigned an IOR or 0.0. A termination surface would have an IOR of -1.0 and typical materials would be assigned an IOR > 1.0. Note that currently IOR values can not be greater than 1000.

To create a parabolic mirror with a foal point at [0,0,0] and a focal length of 5 and a dish diameter of 20 'oe.parabolic_mirror' can be used as follows

'parabolic_mirror = oe.parabolic_mirror(focus=[0,0,0],focal_length=5.0,diameter=20.0,reflectivity = 0.98)'

Note that the reflectivity is encoded in the IOR value of the GeoObject in order to keep the raytracer code simple. This is also the reason why IOR > 1000 is not permitted for refractive materials as an IOR = 1000.98 would mean "Mirror with reflectivity 98%" to the raytracer.

Finally, optical elements can be moved and rotated as follows:

'''
parabolic_mirror.rotate(axis="y",angle=-np.pi/2,pivot = (0,0,0,0))
parabolic_mirror.translate([1,1,0])
'''

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

## Prerequisites

To get LightPyCl to work, you will need the following things.

1. [PyOpenCL](http://mathema.tician.de/software/pyopencl/) for the heavy lifting.
Installing instructions for PyOpenCL can be found [here](http://wiki.tiker.net/PyOpenCL/Installation/Linux).

2. [dxfwrite](https://pypi.python.org/pypi/dxfwrite/) for 3D geometry output.

3. python-numpy and python-matplotlib.

4. Some DXF viewer (like [g3dviewer](http://automagically.de/g3dviewer/) or [gCAD3D](http://www.gcad3d.org/)).

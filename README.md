# **Integrator2** library

CUDA-based library for integration of the Newtonian potential and its gradient in 3D. Contents of the repository:
* source code of the shared library;
* test application for integration in 3D;
* examples of meshes in .dat format.

## Library compilation

### Dependencies

- C++ compiler (tested on MS VC++ 2022 and GCC 9.4.0);
- CMake 3.18 or newer;
- CUDA Toolkit (tested on versions 12.1 and 12.3).

### Build process

```
mkdir build
mkdir install
cd build
cmake ..
make install
```

## Test application

Test app reads the mesh and performs calculation over all existing pairs of neighboring cells. Refinement is done for all cells exactly 3 times.

Available command line options:
| Short option | Long option | Purpose |
|--------------|-------------|---------|
| -h           | --help      | Print the help message |
| -f \<filename\> | --meshfile=\<filename\> | Input mesh file name (required argument) |
| -s \<number\> | --scale=\<number\> | Scale factor for the input mesh |
|              | --exportmesh | Export both the original and the refined mesh to the OBJ file |
|              | --exportresults | Export the results of integration into text files (different files for different types of neighbors) |

Example:
```
integrator2test3D.exe -f ..\..\examples\G1.dat --exportresults
```
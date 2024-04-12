# **Integrator2** library

CUDA-based library for integration of the Newtonian potential and its gradient in 3D. Contents of the repository:

* source code of the shared library;
* test application for integration in 3D;
* examples of meshes in .dat format.

Doxygen-generated [documentation](https://andreyypopov.github.io/integrator2/) is available.

## Library compilation

### Dependencies

* C++ compiler (tested on MS VC++ 2022 and GCC 7.4.0 and 9.4.0);
* CMake 3.18 or newer;
* CUDA Toolkit (tested on versions 10.1, 11.6, 12.1 and 12.3).

### Build process

CMake `BUILD_TESTS` option (`ON` by default) controls the build of the test application.

```bash
mkdir build
mkdir install
cd build
cmake ..
make -j install
```

## Test application

Test app reads the mesh and performs calculation over all existing pairs of neighboring cells.

Available command line options:
| Short option | Long option | Purpose |
|--------------|-------------|---------|
| -h           | --help      | Print the help message |
| -f \<filename\> | --meshfile=\<filename\> | Input mesh file name (required argument) |
| -s \<number\> | --scale=\<number\> | Scale factor for the input mesh |
|              | --exporttoobj | Export the original/refined mesh to the OBJ file |
|              | --exporttovtk | Export the original/refined mesh to the VTK (.vtp) file. Number of refinements is saved for each original cell in case of automatic error control |
|              | --exporttocsv | Export the results of integration into csv files (different files for different types of neighbors) |
|              | --exportresults | Export the results of integration into text files (different files for different types of neighbors) |
| -r \<number\> | --refine=\<number\> | Number of refinement iterations for the whole mesh (in case of 0 the original mesh is used) |
| -c            | --checkresults | Check correctness of integration by comparing integrals for (i, j) and (j, i) pairs. Resulting error can be saved together with integral values |

Example:

```bash
integrator2test3D.exe -f ..\..\examples\s5m.dat -s 0.0005 --exporttocsv --exporttovtk
```

## Noticeable examples

List of the examples includes the following (including preferable value of the scale parameter):
| Surface                   | File name   | No. of vertices   | No. of triangles  | Scale  |
|---------------------------|-------------|-------------------|-------------------|--------|
| Sphere                    | G1.dat      | 55                | 112               | 1.0    |
| Fish                      | Fish.dat    | 1603              | 3194              | 100.0  |
| Airfoil                   | Krylo01.dat | 373               | 854               | 1.0    |
| Kettlebell                | Girja.dat   | 318               | 743               | 1.0    |
| Airplane (coarse mesh)    | s5m.dat     | 1004              | 1864              | 0.0005 |
| Airplane (fine mesh)      | s5m2.dat    | 4086              | 7830              | 0.0005 |
| Propeller                 | Vint16k.dat | 8467              | 17774             | 1.0    |

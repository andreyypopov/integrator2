#include "../../src/Mesh3d.cuh"

int main(int argc, char *argv[]){
    if(argc == 1){
        printf("No input file with mesh specified. Exiting\n");
        return EXIT_FAILURE;
    }
    
    const std::string meshfilename = argv[1];
    double scale = 1.0;
    if(argc > 2)
        scale = std::stod(argv[2]);
    
    Mesh3D mesh;
    if(!mesh.loadMeshFromFile(meshfilename, scale))
        return EXIT_FAILURE;

    mesh.prepareMesh();

    return EXIT_SUCCESS;
}
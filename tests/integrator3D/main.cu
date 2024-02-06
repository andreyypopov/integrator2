#include "../../src/Mesh3d.cuh"
#include "../../src/NumericalIntegrator3d.cuh"
#include "../../src/evaluators/evaluatorJ3DK.cuh"

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

    std::vector<Point3> vertices;
    std::vector<int3> cells;

    vertices.resize(mesh.getVertices().size);
    copy_d2h(mesh.getVertices().data, vertices.data(), mesh.getVertices().size);
    cells.resize(mesh.getCells().size);
    copy_d2h(mesh.getCells().data, cells.data(), mesh.getCells().size);

    exportMeshToObj("OriginalMesh.obj", vertices, cells);

    NumericalIntegrator3D numIntegrator(mesh, qf3D13);
    EvaluatorJ3DK evaluator(mesh, numIntegrator);
    evaluator.setFixedRefinementLevel(3);

    evaluator.runAllPairs();

    evaluator.outputResultsToFile(neighbour_type_enum::simple_neighbors);
    evaluator.outputResultsToFile(neighbour_type_enum::attached_neighbors);
    evaluator.outputResultsToFile(neighbour_type_enum::not_neighbors);

    vertices.resize(numIntegrator.getRefinedVertices().size);
    copy_d2h(numIntegrator.getRefinedVertices().data, vertices.data(), numIntegrator.getRefinedVertices().size);
    cells.resize(numIntegrator.getRefinedCells().size);
    copy_d2h(numIntegrator.getRefinedCells().data, cells.data(), numIntegrator.getRefinedCells().size);

    exportMeshToObj("RefinedMesh.obj", vertices, cells);

    return EXIT_SUCCESS;
}
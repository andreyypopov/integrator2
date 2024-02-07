#include <iostream>

#include "../../src/Mesh3d.cuh"
#include "../../src/NumericalIntegrator3d.cuh"
#include "../../src/evaluators/evaluatorJ3DK.cuh"

#include "../../thirdparty/optionparser-1.7/src/optionparser.h"

struct Arg: public option::Arg
{
  static void printError(const char* msg1, const option::Option& opt, const char* msg2)
  {
    fprintf(stderr, "%s", msg1);
    fwrite(opt.name, opt.namelen, 1, stderr);
    fprintf(stderr, "%s", msg2);
  }

  static option::ArgStatus Unknown(const option::Option& option, bool msg)
  {
    if (msg) printError("Unknown option '", option, "'\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus Required(const option::Option& option, bool msg)
  {
    if (option.arg != 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires an argument\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus NonEmpty(const option::Option& option, bool msg)
  {
    if (option.arg != 0 && option.arg[0] != 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires a non-empty argument\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus Numeric(const option::Option& option, bool msg)
  {
    char* endptr = 0;
    if (option.arg != 0 && strtod(option.arg, &endptr)){};
    if (endptr != option.arg && *endptr == 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires a numeric argument\n");
    return option::ARG_ILLEGAL;
  }
};

enum optionIndex { UNKNOWN, HELP, MESHFILENAME, SCALE, EXPORTMESH, EXPORTRESULTS };

const option::Descriptor usage[] = 
{
    { UNKNOWN,          0, "",  "",             Arg::Unknown,   "USAGE: integrator2test3D [options]\n\n"
                                                                "Options:"},
    { HELP,             0, "h", "help",         Arg::None,      "   -h, \t--help  \tPrint usage and exit." },
    { MESHFILENAME,     0, "f", "meshfile",     Arg::NonEmpty,  "   -f <arg>, \t--meshfile=<arg> \tInput mesh file name." },
    { SCALE,            0, "s", "scale",        Arg::Numeric,   "   -s <arg>, \t--scale=<arg> \tMesh scale factor." },
    { EXPORTMESH,       0, "",  "exportmesh",   Arg::None,      "   \t--exportmesh \tExport original and refined meshes to OBJ files."},
    { EXPORTRESULTS,    0, "",  "exportresults",Arg::None,      "   \t--exportresults \tExport results of integration to text files."},
    { 0, 0, 0, 0, 0, 0 }
};

int main(int argc, char *argv[]){
    argc -= 1;
    argv += 1;

    option::Stats   stats(usage, argc, argv);
    option::Option* options = new option::Option[stats.options_max];
    option::Option* buffer = new option::Option[stats.buffer_max];
    option::Parser  parse(usage, argc, argv, options, buffer);

    if(parse.error())
        return EXIT_FAILURE;

    if(options[HELP] || argc == 0){
        option::printUsage(std::cout, usage);
        return EXIT_SUCCESS;
    }
    
    if(!options[MESHFILENAME]){
        printf("No input file with mesh specified. Exiting\n");
        return EXIT_FAILURE;
    }
    
    const std::string meshfilename(options[MESHFILENAME].arg);
    double scale = 1.0;
    if(options[SCALE])
        scale = std::stod(options[SCALE].arg);
    
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

    if(options[EXPORTMESH])
        exportMeshToObj("OriginalMesh.obj", vertices, cells);

    NumericalIntegrator3D numIntegrator(mesh, qf3D13);
    EvaluatorJ3DK evaluator(mesh, numIntegrator);
    evaluator.setFixedRefinementLevel(3);

    evaluator.runAllPairs();

    if(options[EXPORTRESULTS]){
        evaluator.outputResultsToFile(neighbour_type_enum::simple_neighbors);
        evaluator.outputResultsToFile(neighbour_type_enum::attached_neighbors);
        evaluator.outputResultsToFile(neighbour_type_enum::not_neighbors);
    }

    vertices.resize(numIntegrator.getRefinedVertices().size);
    copy_d2h(numIntegrator.getRefinedVertices().data, vertices.data(), numIntegrator.getRefinedVertices().size);
    cells.resize(numIntegrator.getRefinedCells().size);
    copy_d2h(numIntegrator.getRefinedCells().data, cells.data(), numIntegrator.getRefinedCells().size);

    if(options[EXPORTMESH])
        exportMeshToObj("RefinedMesh.obj", vertices, cells);

    return EXIT_SUCCESS;
}
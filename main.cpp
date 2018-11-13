#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <iostream>
#include <string>
#include "utils.h"
#include "Compute_Platform.h"
#include "ComputationGraph.h"
#include "ComputationRuntime.h"
#include "pugixml.hpp"
#include <cassert>

using std::cout;
using std::string;
using std::endl;
int main(int argc, char* argv[])
{
  assert(argc == 4);
  cout << "Number of input files " << argc << endl;
  auto comp_graph_file = getFullPath((string(__TASK_FOLDER)+argv[1]).c_str());
  auto synthesis_file = getFullPath((string(__TASK_FOLDER) + argv[2]).c_str());
  auto schedule_file = getFullPath((string(__TASK_FOLDER) + argv[3]).c_str());
  assert(comp_graph_file.empty ()==false);
  assert(synthesis_file.empty ()==false);
  assert(schedule_file.empty ()==false);

  Compute_Platform test = Compute_Platform(synthesis_file);
  ComputationGraph graph = ComputationGraph(comp_graph_file,test);
  ComputationRuntime runtime = ComputationRuntime(test);
  bool match = runtime.execute_compute_graph(graph);
  cout << "Results " << (match ? "matched" : "mismatched") << endl;
  cout<<"Press enter to continue"<<endl;
  std::cin.get ();
  exit(match ? EXIT_SUCCESS : EXIT_FAILURE);
}
#pragma once
#include <tuple>
#include <string>
#include <vector>
#include <typeinfo>
#include "ComputationNode.hpp"
#include "pugixml.hpp"
#include <map>
using std::vector;
using std::string;
using std::map;
enum node_t {producer,consumer,worker,none};
enum src_trg {src= true,target=false};
typedef std::tuple<std::string, src_trg> connection;
struct graph_node
{
	node_t type = none;
	string dtype ="";
	string name = "";
	string subtype = "";
	int dimension[3];
	vector<connection> connections;	
};
class CompGraphParser
{
	map<string, graph_node> worker_nodes;	
	map<string, graph_node> producer_consumer_nodes;
	
public:
	CompGraphParser() = default;
	CompGraphParser(pugi::xml_document& doc, map<string, vector<int>>& fifos, map<string, cl::Buffer>& buffer, const cl::Context& ctx);
	std::map<string,graph_node> get_worker_nodes() const;
	void set_comp_node_data_buffers(ComputationNode& computation_node, string kernel_name, map<string, vector<int>>& fifos, map<string, cl::Buffer>& buffer);
	~CompGraphParser();
};


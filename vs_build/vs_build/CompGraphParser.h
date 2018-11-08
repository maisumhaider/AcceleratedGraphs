#pragma once
#include <tuple>
#include <string>
#include <vector>
#include <typeinfo>
#include "ComputationNode.hpp"
#include "../../pugixml-1.9/pugixml-1.9/src/pugixml.hpp"
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
	CompGraphParser(pugi::xml_document& doc);
	std::map<string,graph_node> get_worker_nodes() const;
	void set_comp_node_data_buffers(ComputationNode& computation_node, string kernel_name);
	~CompGraphParser();
};


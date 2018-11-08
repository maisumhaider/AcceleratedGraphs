#include "CompGraphParser.h"
#include <cassert>


node_t get_type(string type)
{
	if (type == "producer") return producer;
	if (type == "consumer") return consumer;
	if (type == "worker") return worker;
	return none;
}

void dimensionalityCheck(int arr[3])
{
	assert(arr[2] > 0);
	assert(arr[1] > 0);
	assert(arr[0] > 0);
}
CompGraphParser::CompGraphParser(pugi::xml_document& doc)
{
	pugi::xml_node xml_nodes = doc.child("graph").child("nodes");
	pugi::xml_node connections = doc.child("graph").child("connections");
	for (pugi::xml_node xml_node_ = xml_nodes.first_child(); xml_node_; xml_node_ = xml_node_.next_sibling())
	{
		graph_node node;
		node.type = get_type(xml_node_.attribute("type").value());
		node.name = xml_node_.attribute("name").value();		
		if (node.type == producer || node.type == consumer){
			node.dtype = xml_node_.attribute("dtype").value();
			node.dimension[0] = xml_node_.child("dimension").child("x").text().as_int();
			node.dimension[1] = xml_node_.child("dimension").child("y").text().as_int();
			node.dimension[2] = xml_node_.child("dimension").child("z").text().as_int();
			dimensionalityCheck(node.dimension);
			producer_consumer_nodes[node.name] = node;
		}
		else if (node.type == worker)
		{
			node.subtype = xml_node_.attribute("subtype").value();
			for (pugi::xml_node node_conn = connections.first_child(); node_conn; node_conn = node_conn.next_sibling())
			{
				string conn_src = node_conn.attribute("source").value();
				string conn_target = node_conn.attribute("target").value();
				if (node.name == conn_src){
					connection val = connection(conn_target, target);
					node.connections.push_back(val);
				}
				else if (node.name == conn_target)
				{
					connection val = connection(conn_src,src);
					node.connections.push_back(val);
				}
			}
			worker_nodes[node.name] = node;
		}
	}
}

std::map<string,graph_node> CompGraphParser::get_worker_nodes() const
{
	return worker_nodes;
}


void CompGraphParser::set_comp_node_data_buffers(ComputationNode& computation_node, string kernel_name)
{
	graph_node worker = worker_nodes[kernel_name];
	int prevDim[3] = {0};
	bool first = true;
	for (auto connection : worker.connections)
	{
		string conn_node = std::get<0>(connection);
		bool is_src = std::get<1>(connection);
		auto prod_con = producer_consumer_nodes[conn_node];
		
		//Note The following is a check to ensure that all input and output of a worker node are same. 
		// Assumption: Homogenous Static Dataflow where each node consumes one node from every input and produces one token on output.
		if (first)
		{
			prevDim[0] = prod_con.dimension[0];
			prevDim[1] = prod_con.dimension[1];
			prevDim[2] = prod_con.dimension[2];
		}
		else
		{
			assert(prevDim[0] == prod_con.dimension[0]);
			assert(prevDim[1] == prod_con.dimension[1]);
			assert(prevDim[2] == prod_con.dimension[2]);
		}
		first = false;
		const auto flatten_dim = prod_con.dimension[0] * prod_con.dimension[1] * prod_con.dimension[2];
		if (is_src)
		{
			computation_node.add_input_data_buffer(vector<int>(flatten_dim,1));
		}
		else {
			computation_node.add_output_data_buffer(vector<int>(flatten_dim,0));
		}
	}

}

CompGraphParser::~CompGraphParser()
{
}

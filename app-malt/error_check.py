import sys
import json
import traceback
import re
import numpy as np
import pandas as pd
import networkx as nx


class SafetyChecker():
    def __init__(self, ret_graph=None, ret_list=None):
        if ret_graph:
            self.graph = ret_graph
        else:
            self.graph = None

    def evaluate_all(self):
        if self.graph:
            graph_checks = [self.verify_node_format_and_type,
                            self.verify_edge_format_and_type,
                            self.verify_node_hierarchy,
                            self.verify_no_isolated_nodes,
                            self.verify_bandwidth, 
                            self.verify_port_exist]
            for check in graph_checks:
                try:
                    success, message = check()
                    if not success:
                        return False, message
                except Exception as e:
                    print("Check failed:", e)
                    print(traceback.format_exc())
                    return False, str(e)
            return True, ""

    def verify_node_format_and_type(self):
        """
        Graph check: verify node type and format
        """
        valid_types = ['EK_SUPERBLOCK', 'EK_CHASSIS', 'EK_RACK', 'EK_AGG_BLOCK', 'EK_JUPITER', 'EK_PORT', 'EK_SPINEBLOCK', 'EK_PACKET_SWITCH', 'EK_CONTROL_POINT', 'EK_CONTROL_DOMAIN']

        for node in self.graph.nodes():
            # Check if the node has a 'type' attribute
            if not self.graph.nodes[node].get('type'):
                return False, f"Node '{node}' is missing the required 'type' attribute"
            
            node_types = self.graph.nodes[node]['type']
            # Handle both single type (string) and multiple types (list) cases
            if isinstance(node_types, str):
                node_types = [node_types]
            
            for node_type in node_types:
                if node_type not in valid_types:
                    return False, f"Node '{node}' has invalid type '{node_type}'. Valid types are: {', '.join(valid_types)}"

        return True, ""

    def verify_edge_format_and_type(self):
        """
        Graph check: verify_edge_format_and_type
        """
        valid_edge_types = ["RK_CONTAINS", "RK_CONTROLS"]

        for edge in self.graph.edges(data=True):
            # Check if the edge has a 'type' attribute
            if 'type' not in edge[2]:
                return False, f"Edge between '{edge[0]}' and '{edge[1]}' is missing the required 'type' attribute"
            # Check if the edge's type is in the valid_edge_types list
            if not any(edge_type in edge[2]['type'] for edge_type in valid_edge_types):
                return False, f"Edge between '{edge[0]}' and '{edge[1]}' has invalid type '{edge[2]['type']}'. Valid types are: {', '.join(valid_edge_types)}"
        
        # Only return True if all edges have passed the checks
        return True, ""

    def verify_node_hierarchy(self):
        """
        Graph check: verify_node_hierarchy
        """
        hierarchy = {
            "EK_JUPITER": ["EK_SPINEBLOCK", "EK_SUPERBLOCK"],
            "EK_SPINEBLOCK": ["EK_PACKET_SWITCH"],
            "EK_SUPERBLOCK": ["EK_AGG_BLOCK"],
            "EK_AGG_BLOCK": ["EK_PACKET_SWITCH"],
            "EK_CHASSIS": ["EK_CONTROL_POINT", "EK_PACKET_SWITCH"],
            "EK_CONTROL_POINT": ["EK_PACKET_SWITCH"],
            "EK_RACK": ["EK_CHASSIS"],
            "EK_PACKET_SWITCH": ["EK_PORT"],
            "EK_CONTROL_DOMAIN": ["EK_CONTROL_POINT", "EK_PACKET_SWITCH"]
        }

        for edge in self.graph.edges(data=True):
            if 'RK_CONTAINS' in edge[2]['type']:
                source_node = self.graph.nodes[edge[0]].get('name', edge[0])
                target_node = self.graph.nodes[edge[1]].get('name', edge[1])
                source_node_types = self.graph.nodes[edge[0]]['type']
                target_node_types = self.graph.nodes[edge[1]]['type']
                # Convert to list if string
                if isinstance(source_node_types, str):
                    source_node_types = [source_node_types]
                if isinstance(target_node_types, str):
                    target_node_types = [target_node_types]
                
                # Check if any valid hierarchy relationship exists between any source and target types
                valid_hierarchy = any(
                    source_type in hierarchy and
                    any(t_type in hierarchy[source_type] for t_type in target_node_types)
                    for source_type in source_node_types
                )
                
                if not valid_hierarchy:
                    return False, f"Invalid hierarchy: node '{source_node}' of type(s) '{source_node_types}' cannot contain node '{target_node}' of type(s) '{target_node_types}'"
        
        return True, ""
    
    def verify_no_isolated_nodes(self):
        """
        Graph check: verify_no_isolated_nodes
        """
        # An isolated node is a node with degree 0, i.e., no edges.
        isolated_nodes = list(nx.isolates(self.graph))

        if len(isolated_nodes) == 0:
            return True, ""  # There are no isolated nodes in the graph.
        else:
            return False, f"Found {len(isolated_nodes)} isolated nodes: {', '.join(str(node) for node in isolated_nodes)}"


    def verify_bandwidth(self):
        """
        Verify if all ports in the graph have non-zero bandwidth.

        Returns:
            tuple: (bool, str) - (True, "") if all ports have valid bandwidth, 
                                (False, error_message) otherwise.
        """
        for node in self.graph.nodes():
            if 'EK_PORT' in self.graph.nodes[node]['type']:
                if 'physical_capacity_bps' not in self.graph.nodes[node]:
                    return False, f"Port node '{node}' is missing the required 'physical_capacity_bps' attribute"
                if self.graph.nodes[node]['physical_capacity_bps'] == 0:
                    return False, f"Port node '{node}' has invalid bandwidth of 0 bps"
        
        return True, ""

    def verify_port_exist(self):
        """
        Verify with the given graph, for all nodes with type EK_PACKET_SWITCH, check if it has at least one port with type EK_PORT.
        """
        for node in self.graph.nodes():
            node_types = self.graph.nodes[node]['type']
            if 'EK_PACKET_SWITCH' in node_types:
                port_count = sum(1 for neighbor in self.graph.successors(node) 
                               if 'EK_PORT' in self.graph.nodes[neighbor]['type'])
                if port_count == 0:
                    return False, f"Packet switch node '{node}' has no ports connected to it"
        
        return True, ""

    #TODO: add each port can only be connected with one packet switch


#ifndef KD_TREE_H
#define KD_TREE_H

#include "variables.h"

void create_kd_tree(scene* scn) ;

void create_kd_tree(sse_scene* scn) ;

void destroy_kd_tree(kd_tree_node* kd_tree) ;

bound_box kd_find_leaf(kd_tree_node* kd_tree, float x, float y, float z) ;

#endif


#include <vector>
#include "variables.h"

// splits the contents of in into a low and high box by the median of the tops of the boxes
float split_x_median(vector<bound_box>& in, vector<bound_box>& low, vector<bound_box>& high) {

	unsigned int l = 0, h = in.size()-1;
	unsigned int l2, h2;
	float mid;

	while ( l < in.size()/2 && h > in.size()/2 ) {

		l2 = l;
		h2 = h;

		mid = in[in.size()/2].top_x;

		while( l2 < h2 ) {

			while ( in[l2].top_x < mid && l2 < h2 )
				++l2;
			while ( in[h2].top_x > mid && l2 < h2 )
				--h2;

			bound_box tmp = in[l2];
			in[l2] = in[h2];
			in[h2] = tmp;

			if ( l2 < h2 ) {
				++l2;
				--h2;
			}
		}

		if (h2 > in.size()/2)
			h = h2;
		else l = l2;

	}

	mid = in[in.size()/2].top_x + SMALL;

	for (unsigned int i = 0; i < in.size(); ++i) {

		if (in[i].bot_x <= mid)
			low.push_back(in[i]);
		if (in[i].top_x >= mid)
			high.push_back(in[i]);
	}

	return mid;
}

float split_y_median(vector<bound_box>& in, vector<bound_box>& low, vector<bound_box>& high) {

	unsigned int l = 0, h = in.size()-1;
	unsigned int l2 = l, h2 = h;
	float mid;

	while ( l < in.size()/2 && h > in.size()/2 ) {

		l2 = l;
		h2 = h;

		mid = in[in.size()/2].top_y;

		while( l2 < h2 ) {

			while ( in[l2].top_y < mid && l2 < h2 )
				++l2;
			while ( in[h2].top_y > mid && l2 < h2 )
				--h2;

			bound_box tmp = in[l2];
			in[l2] = in[h2];
			in[h2] = tmp;

			if ( l2 < h2 ) {
				++l2;
				--h2;
			}
		}

		if (h2 > in.size()/2)
			h = h2;
		else l = l2;

	}

	mid = in[in.size()/2].top_y + SMALL;

	for (unsigned int i = 0; i < in.size(); ++i) {

		if (in[i].bot_y <= mid)
			low.push_back(in[i]);
		if (in[i].top_y >= mid)
			high.push_back(in[i]);
	}

	return mid;
}

float split_z_median(vector<bound_box>& in, vector<bound_box>& low, vector<bound_box>& high) {

	unsigned int l = 0, h = in.size()-1;
	unsigned int l2 = l, h2 = h;
	float mid;

	while ( l < in.size()/2 && h > in.size()/2 ) {

		l2 = l;
		h2 = h;

		mid = in[in.size()/2].top_z;

		while( l2 < h2 ) {

			while ( in[l2].top_z < mid && l2 < h2 )
				++l2;
			while ( in[h2].top_z > mid && l2 < h2 )
				--h2;

			bound_box tmp = in[l2];
			in[l2] = in[h2];
			in[h2] = tmp;

			if ( l2 < h2 ) {
				++l2;
				--h2;
			}
		}

		if (h2 > in.size()/2) 
			h = h2;
		else 
			l = l2;

	}

	mid = in[in.size()/2].top_z + SMALL;

	for (unsigned int i = 0; i < in.size(); ++i) {

		if (in[i].bot_z <= mid)
			low.push_back(in[i]);
		if (in[i].top_z >= mid)
			high.push_back(in[i]);
	}

	return mid;
}

// finds the distance between the furthest spread points
float find_x_span (vector<bound_box>& in) {

	if (in.size() == 0) return 0.0f;

	float low = INF, high = -INF;

	for (unsigned int i = 0; i < in.size(); ++i) {

		if (in[i].top_x < low)
			low = in[i].top_x;
		if (in[i].top_x > high)
			high = in[i].top_x;
	}

	return high - low;
}

float find_y_span (vector<bound_box>& in) {

	if (in.size() == 0) return 0.0f;

	float low = INF, high = -INF;

	for (unsigned int i = 0; i < in.size(); ++i) {

		if (in[i].top_y < low)
			low = in[i].top_y;
		if (in[i].top_y > high)
			high = in[i].top_y;
	}

	return high - low;
}

float find_z_span (vector<bound_box>& in) {

	if (in.size() == 0) return 0.0f;

	float low = INF, high = -INF;

	for (unsigned int i = 0; i < in.size(); ++i) {

		if (in[i].top_z < low)
			low = in[i].top_z;
		if (in[i].top_z > high)
			high = in[i].top_z;
	}

	return high - low;
}

void make_kd_leaf(kd_tree_node* node, vector<bound_box>& boxes) {

	// make leaf
	node->type = LEAF;

	// prepare leaf's array
	node->items = (unsigned int *) malloc((boxes.size()+3)*sizeof(unsigned int));
	
	node->items[0] = boxes.size()+4;

	unsigned int i = 4;

	// add triangles
	for (unsigned int j = 0; j < boxes.size(); ++j ) {
		if (boxes[j].type != TRIANGLE && boxes[j].type != SPHERE && boxes[j].type != ARBITRARY_SPHERE) {
			printf("bad box");
		}
		else if (boxes[j].type == TRIANGLE) {
			node->items[i] = boxes[j].obj;
			++i;
		}
	}

	// parallelograms offset
	node->items[1] = i;

	// add parallelograms
	for (unsigned int j = 0; j < boxes.size(); ++j ) {
		if (boxes[j].type == PARALLELOGRAM) {
			node->items[i] = boxes[j].obj;
			++i;
		}
	}

	// spheres offset
	node->items[2] = i;

	// add spheres
	for (unsigned int j = 0; j < boxes.size(); ++j) {
		if (boxes[j].type == SPHERE) {
			node->items[i] = boxes[j].obj;
			++i;
		}
	}

	// arbitrary spheres offset
	node->items[3] = i;

	// add spheres
	for (unsigned int j = 0; j < boxes.size(); ++j) {
		if (boxes[j].type == ARBITRARY_SPHERE) {
			node->items[i] = boxes[j].obj;
			++i;
		}
	}

	//printf("leaf %d\n", boxes.size());
}

void create_kd_tree_node(unsigned int pos, unsigned int depth, vector<bound_box>& boxes, kd_tree_node* kd_tree, unsigned int max_kd_depth, unsigned int min_kd_leaf_size) {

	if (depth >= max_kd_depth || boxes.size() < min_kd_leaf_size) {
		
		// make leaf
		make_kd_leaf(&kd_tree[pos], boxes);
	} 
	else {
		
		float x_span = find_x_span(boxes);
		float y_span = find_y_span(boxes);
		float z_span = find_z_span(boxes);

		vector<bound_box> low;
		vector<bound_box> high;

		unsigned char axis = 0xff;
		float mid;

		// split based upon which direction has the greatest spread
		if (x_span >= y_span) {
			if (x_span >= z_span) {
			
				axis = X;
				mid = split_x_median (boxes, low, high);
			} else {

				axis = Z;
				mid = split_z_median (boxes, low, high);
			}
		}
		else {
			if (y_span >= z_span) {

				axis = Y;
				mid = split_y_median (boxes, low, high);
			}
			else {

				axis = Z;
				mid = split_z_median (boxes, low, high);
			}
		}
		
		/*
		// slower
		vector<bound_box> low;
		vector<bound_box> high;
		vector<bound_box> lowX;
		vector<bound_box> highX;
		vector<bound_box> lowY;
		vector<bound_box> highY;
		vector<bound_box> lowZ;
		vector<bound_box> highZ;

		float midX = split_x_median (boxes, lowX, highX);
		float midY = split_y_median (boxes, lowY, highY);
		float midZ = split_z_median (boxes, lowZ, highZ);

		int x_border = highX.size() + lowX.size() - boxes.size();
		int y_border = highY.size() + lowY.size() - boxes.size();
		int z_border = highZ.size() + lowZ.size() - boxes.size();

		unsigned char axis = 0xff;
		float mid;

		// split based upon which direction has the greatest spread
		if (x_border <= y_border) {
			if (x_border <= z_border) {
			
				axis = X;
				mid = midX;
				low = lowX;
				high = highX;
			} else {

				axis = Z;
				mid = midZ;
				low = lowZ;
				high = highZ;
			}
		}
		else {
			if (y_border <= z_border) {

				axis = Y;
				mid = midY;
				low = lowY;
				high = highY;
			}
			else {

				axis = Z;
				mid = midZ;
				low = lowZ;
				high = highZ;
			}
		}
		*/

		// check split was effective (4/5 is arbitrary)
		if (low.size() < 4*boxes.size()/5 && high.size() < 4*boxes.size()/5) {

			// make non-leaf
			kd_tree[pos].type = axis;
			kd_tree[pos].split = mid;

			//printf("low %d, mid %0.3f, axis %s\n", low.size(), mid, axis == X ? "x" : axis == Y ? "y" : "z");
			create_kd_tree_node(pos*2, depth+1, low, kd_tree, max_kd_depth, min_kd_leaf_size);

			//printf("high %d, mid %0.3f, axis %s\n", high.size(), mid, axis == X ? "x" : axis == Y ? "y" : "z");
			create_kd_tree_node(pos*2+1, depth+1, high, kd_tree, max_kd_depth, min_kd_leaf_size);
		} 
		else {

			// make leaf
			make_kd_leaf(&kd_tree[pos], boxes);
		}
	}
}

void create_kd_tree(scene* scn) {

	// set up vector with available nodes
	scn->kd_tree = (kd_tree_node*) malloc( ((int)pow(2.0f, (int)(scn->max_kd_depth+1)) + 1 ) * sizeof(kd_tree_node));

	create_kd_tree_node(1, 0, scn->boxes, scn->kd_tree, scn->max_kd_depth, scn->min_kd_leaf_size);

	printf("Sucessfully created kd-tree\n");
}

void create_kd_tree(sse_scene* scn) {

	// set up vector with available nodes
	scn->kd_tree = (kd_tree_node*) malloc( ((int)pow(2.0f, (int)(scn->max_kd_depth+1)) + 1 ) * sizeof(kd_tree_node));
	if (scn->kd_tree == NULL) {
		printf("Error, malloc failed\n");
		exit(1);
	}

	create_kd_tree_node(1, 0, scn->boxes, scn->kd_tree, scn->max_kd_depth, scn->min_kd_leaf_size);

	printf("Sucessfully created kd-tree\n");
}

void destroy_kd_node(kd_tree_node* kd_tree, int pos) {

	if (kd_tree[pos].type == LEAF) {
		free(kd_tree[pos].items);
	} else {
		destroy_kd_node(kd_tree, pos*2);
		destroy_kd_node(kd_tree, pos*2 + 1);
	}
}

void destroy_kd_tree(kd_tree_node* kd_tree) {

	// free arrays at leaves
	destroy_kd_node(kd_tree, 1);

	// free tree itself
	free(kd_tree);
}


bound_box kd_find_leaf(kd_tree_node* kd_tree, float x, float y, float z) {

	// start at the top of the tree
	int kd_pos = 1;

	// start with a INF/infinite size box
	bound_box a = {KD_BOX, 0, INF, -INF, INF, -INF, INF, -INF};

	kd_tree_node* here = &kd_tree[kd_pos];

	while (here->type != LEAF) {

		// move to left child
		kd_pos *= 2;

		// successively bind bound box more tightly as move down tree
		if (here->type == X) {

			if (here->split < x) { // go to right child
				kd_pos += 1;
				a.bot_x = here->split; // bound below
			} 
			else { // stay at left child
				a.top_x = here->split; // bound above
			}
		}
		else if (here->type == Y) {

			if (here->split < y) { // go to right child
				kd_pos += 1;
				a.bot_y = here->split; // bound below
			} 
			else { // stay at left child
				a.top_y = here->split; // bound above
			}
		}
		else if (here->type == Z) {

			if (here->split < z) { // go to right child
				kd_pos += 1;
				a.bot_z = here->split; // bound below
			} 
			else { // stay at left child
				a.top_z = here->split; // bound above
			}
		}

		// move node pointer
		here = &kd_tree[kd_pos];
	}
	a.obj = kd_pos; // set index into kd_tree

	//printf("found leaf at %d\n", kd_pos);

	return a;
}


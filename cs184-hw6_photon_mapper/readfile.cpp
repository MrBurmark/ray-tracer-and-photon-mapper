
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <deque>
#include <stack>

#include "variables.h"
#include "kd_tree.h"


// Function to read the input data values
bool readvals(stringstream &s, const int numvals, float * values) {
	for (int i = 0 ; i < numvals ; i++) {
		s >> values[i] ; 
		if (s.fail()) {
			cout << "Failed reading value " << i << " will skip\n" ; 
			return false ;
		}
	}
	return true ; 
}

// Function to read the input data values
bool readintvals(stringstream &s, const int numvals, int * values) {
	for (int i = 0 ; i < numvals ; i++) {
		s >> values[i] ; 
		if (s.fail()) {
			cout << "Failed reading value " << i << " will skip\n" ; 
			return false ;
		}
	}
	return true ; 
}


void norm_triangle(triangle* tri, triangle_property* triprop) {

	vec3 norm = glm::normalize(glm::cross(tri->T1, tri->T2));
	triprop->norm0 = norm;
	triprop->norm1 = norm;
	triprop->norm2 = norm;
}

void perp_triangle(triangle* tri, triangle_property* triprop) {

	vec3 norm = glm::normalize(glm::cross(tri->T1, tri->T2));
	triprop->norm = norm;
	triprop->perp0 = glm::normalize(glm::cross(norm, (tri->T1 - tri->T2)));
	triprop->perp1 = glm::normalize(glm::cross(norm, tri->T2));
	triprop->perp2 = glm::normalize(glm::cross(tri->T1, norm));
}

void norm_parallelogram(parallelogram* par, parallelogram_property* parprop) {

	vec3 norm = glm::normalize(glm::cross(par->T1, par->T2));
	parprop->norm0 = norm;
	parprop->norm1 = norm;
	parprop->norm2 = norm;
	parprop->norm3 = norm;
}

void perp_parallelogram(parallelogram* par, parallelogram_property* parprop) {

	vec3 norm = glm::normalize(glm::cross(par->T1, par->T2));
	parprop->norm = norm;
	parprop->perp1 = glm::normalize(glm::cross(par->T1, norm));
	parprop->perp2 = glm::normalize(glm::cross(norm, par->T2));
}

void make_rec_light(rec_light* rec, float loc_radius) {

	float W = glm::length(rec->width);
	unsigned int w_size = (unsigned int)ceilf(W/loc_radius);

	float H = glm::length(rec->height);
	unsigned int h_size = (unsigned int)ceilf(H/loc_radius);

	unsigned int w;
	unsigned int h;
	vec3 pt(0.0f);

	for (w = 0; w <= w_size; ++w) {

		for (h = 0; h <= h_size; ++h) {

			pt = rec->corner + rec->width * (w/(float)w_size) + rec->height * (h/(float)h_size);

			rec->pts.push_back(pt);
		}
	}
}

void readfile(const char * filename, scene* scn) {
	string str, cmd ; 
	ifstream in ;
	in.open(filename) ; 
	if (in.is_open()) {

		printf("Reading File\n");

		// matrix stack to store transforms.  
		stack <mat4_ex> transfstack ; 
		mat4_ex mat1 = {mat4(1.0), false};
		transfstack.push( mat1 ) ; // identity

		stack <mat4> invtransptransfstack ; 
		invtransptransfstack.push(mat4(1.0f));

		vector <vec3> vertices ; 

		vector <vec3> vertexnormals ; 
		vector <vec3> normals ; 

		vec3 ambient(0.2f);
		vec3 diffuse(0.0f);
		vec3 specular(0.0f);
		vec3 emission(0.0f);
		float shininess = 0.0f;
		float alpha = 1.0f;
		float refraction_index = 1.0f;

		getline (in, str) ; 
		while (in) {
			if ((str.find_first_not_of(" \t\r\n") != string::npos) && (str[0] != '#') && !(str[str.find_first_not_of(" \t\r\n")] == '/' && str[str.find_first_not_of(" \t\r\n")+1] == '/')) {
				// Ruled out comment and blank lines 

				stringstream s(str) ;
				s >> cmd ; 

				float values[16] ;  
				int intvalues[16] ;  
				bool validinput ;

				if (cmd == "vertex") {
					validinput = readvals(s, 3, values) ; 
					if (validinput) {
						vertices.push_back(vec3(values[0], values[1], values[2]));
					}			
				}

				else if (cmd == "vertexnormal") {
					validinput = readvals(s, 6, values) ; 
					if (validinput) {
						vertexnormals.push_back(vec3(values[0], values[1], values[2]));
						normals.push_back(vec3(values[3], values[4], values[5]));
					}			
				}

				else if (cmd == "tri") {
					validinput = readintvals(s, 3, intvalues) ; 
					if (validinput) {

						triangle tri = {};
						triangle_property triprop = {};

						// transform points into world coordinates
						vec3 pt0 = vertices[intvalues[0]];
						pt0 = vec3(transfstack.top().mat * vec4(pt0, 1.0f));
						tri.pt0 = pt0;


						vec3 pt1 = vertices[intvalues[1]];
						pt1 = vec3(transfstack.top().mat * vec4(pt1, 1.0f));
						tri.T1 = pt1 - pt0;


						vec3 pt2 = vertices[intvalues[2]];
						pt2 = vec3(transfstack.top().mat * vec4(pt2, 1.0f));
						tri.T2 = pt2 - pt0;

						// get normals in world coordinates
						norm_triangle(&tri, &triprop);

						perp_triangle(&tri, &triprop);

						triprop.ambient = ambient;
						triprop.emission = emission;
						triprop.diffuse = diffuse ; 
						triprop.specular = specular ; 
						triprop.shininess = shininess ; 
						triprop.index = refraction_index ;
						triprop.alpha = alpha ;

						scn->triangles.push_back(tri) ; 
						scn->triangle_properties.push_back(triprop);
					}			
				}

				else if (cmd == "trinormal") {
					validinput = readintvals(s, 6, intvalues) ; 
					if (validinput) {

						triangle tri = {};
						triangle_property triprop = {};

						// transform into world coordinates
						vec3 pt0 = vertexnormals[intvalues[0]];
						pt0 = vec3(transfstack.top().mat * vec4(pt0, 1.0f));
						tri.pt0 = pt0;

						vec3 pt1 = vertexnormals[intvalues[1]];
						pt1 = vec3(transfstack.top().mat * vec4(pt1, 1.0f));
						tri.T1 = pt1 - pt0;

						vec3 pt2 = vertexnormals[intvalues[2]];
						pt2 = vec3(transfstack.top().mat * vec4(pt2, 1.0f));
						tri.T2 = pt2 - pt0;

						// transform normals into world coordinates with inverse transpose
						pt0 = normals[intvalues[3]];
						triprop.norm0 = vec3(invtransptransfstack.top() * vec4(pt0.x, pt0.y, pt0.z, 0.0f));

						pt1 = normals[intvalues[4]];
						triprop.norm1 = vec3(invtransptransfstack.top() * vec4(pt1.x, pt1.y, pt1.z, 0.0f));

						pt2 = normals[intvalues[5]];
						triprop.norm2 = vec3(invtransptransfstack.top() * vec4(pt2.x, pt2.y, pt2.z, 0.0f));


						perp_triangle(&tri, &triprop);

						triprop.ambient = ambient;
						triprop.emission = emission;
						triprop.diffuse = diffuse ; 
						triprop.specular = specular ; 
						triprop.shininess = shininess ; 
						triprop.index = refraction_index ;
						triprop.alpha = alpha ;

						scn->triangles.push_back(tri);
						scn->triangle_properties.push_back(triprop);
					}			
				}

				else if (cmd == "par") {
					validinput = readintvals(s, 3, intvalues) ; 
					if (validinput) {

						parallelogram par = {};
						parallelogram_property parprop = {};

						// transform points into world coordinates
						vec3 pt0 = vertices[intvalues[0]];
						pt0 = vec3(transfstack.top().mat * vec4(pt0, 1.0f));
						par.pt0 = pt0;

						vec3 pt1 = vertices[intvalues[1]];
						pt1 = vec3(transfstack.top().mat * vec4(pt1, 1.0f));
						par.T1 = pt1 - pt0;


						vec3 pt2 = vertices[intvalues[2]];
						pt2 = vec3(transfstack.top().mat * vec4(pt2, 1.0f));
						par.T2 = pt2 - pt0;

						// get normals in world coordinates
						norm_parallelogram(&par, &parprop);

						perp_parallelogram(&par, &parprop);

						parprop.ambient = ambient;
						parprop.emission = emission;
						parprop.diffuse = diffuse ; 
						parprop.specular = specular ; 
						parprop.shininess = shininess ; 
						parprop.index = refraction_index ;
						parprop.alpha = alpha ;

						scn->parallelograms.push_back(par) ; 
						scn->parallelogram_properties.push_back(parprop);
					}			
				}

				else if (cmd == "parnormal") {
					validinput = readintvals(s, 7, intvalues) ; 
					if (validinput) {

						parallelogram par = {};
						parallelogram_property parprop = {};

						// transform into world coordinates
						vec3 pt0 = vertexnormals[intvalues[0]];
						pt0 = vec3(transfstack.top().mat * vec4(pt0, 1.0f));
						par.pt0 = pt0;

						vec3 pt1 = vertexnormals[intvalues[1]];
						pt1 = vec3(transfstack.top().mat * vec4(pt1, 1.0f));
						par.T1 = pt1 - pt0;

						vec3 pt2 = vertexnormals[intvalues[2]];
						pt2 = vec3(transfstack.top().mat * vec4(pt2, 1.0f));
						par.T2 = pt2 - pt0;

						// transform normals into world coordinates with inverse transpose
						pt0 = normals[intvalues[3]];
						parprop.norm0 = vec3(invtransptransfstack.top() * vec4(pt0.x, pt0.y, pt0.z, 0.0f));

						pt1 = normals[intvalues[4]];
						parprop.norm1 = vec3(invtransptransfstack.top() * vec4(pt1.x, pt1.y, pt1.z, 0.0f));

						pt2 = normals[intvalues[5]];
						parprop.norm2 = vec3(invtransptransfstack.top() * vec4(pt2.x, pt2.y, pt2.z, 0.0f));

						vec3 pt3 = normals[intvalues[6]];
						parprop.norm3 = vec3(invtransptransfstack.top() * vec4(pt2.x, pt2.y, pt2.z, 0.0f));


						perp_parallelogram(&par, &parprop);

						parprop.ambient = ambient;
						parprop.emission = emission;
						parprop.diffuse = diffuse ; 
						parprop.specular = specular ; 
						parprop.shininess = shininess ; 
						parprop.index = refraction_index ;
						parprop.alpha = alpha ;

						scn->parallelograms.push_back(par);
						scn->parallelogram_properties.push_back(parprop);
					}			
				}

				else if (cmd == "sphere") {
					validinput = readvals(s, 4, values) ; 
					if (validinput) {
						if (transfstack.top().non_uniform_scale) {

							arbitrary_sphere sph = {};
							sphere_property sphprop = {};

							sph.pos = vec3(values[0], values[1], values[2]); 
							sph.radius2 = values[3]*values[3];
							sph.inversetransform = glm::transpose(invtransptransfstack.top());

							sphprop.ambient = ambient;
							sphprop.emission = emission;
							sphprop.diffuse = diffuse ; 
							sphprop.specular = specular ; 
							sphprop.shininess = shininess ; 
							sphprop.index = refraction_index ;
							sphprop.alpha = alpha ;

							scn->arbitrary_spheres.push_back(sph) ; 
							scn->arbitrary_sphere_properties.push_back(sphprop);
						}
						else {

							sphere sph = {};
							sphere_property sphprop = {};

							vec3 center = vec3(transfstack.top().mat * vec4(values[0], values[1], values[2], 1.0f));
							vec3 radius = vec3(transfstack.top().mat * vec4(values[0], values[1], values[2] + values[3], 1.0f)) - center;

							sph.pos = center;
							sph.radius2 = glm::dot(radius, radius);

							sphprop.ambient = ambient;
							sphprop.emission = emission;
							sphprop.diffuse = diffuse ; 
							sphprop.specular = specular ; 
							sphprop.shininess = shininess ; 
							sphprop.index = refraction_index ;
							sphprop.alpha = alpha ;

							scn->spheres.push_back(sph) ; 
							scn->sphere_properties.push_back(sphprop);
						}
					}
				}

				else if (cmd == "alpha") {
					validinput = readvals(s, 1, values) ; 
					if (validinput) 
						alpha = values[0]; 
				}
				else if (cmd == "index") {
					validinput = readvals(s, 1, values) ; 
					if (validinput) 
						refraction_index = values[0]; 
				}
				else if (cmd == "ambient") {
					validinput = readvals(s, 3, values) ; 
					if (validinput) 
						ambient = vec3(values[0], values[1], values[2]); 
				}
				else if (cmd == "diffuse") {
					validinput = readvals(s, 3, values) ; 
					if (validinput) 
						diffuse = vec3(values[0], values[1], values[2]); 
				}
				else if (cmd == "specular") {
					validinput = readvals(s, 3, values) ; 
					if (validinput) 
						specular = vec3(values[0], values[1], values[2]); 
				}
				else if (cmd == "emission") {
					validinput = readvals(s, 3, values) ; 
					if (validinput) 
						emission = vec3(values[0], values[1], values[2]); 
				}
				else if (cmd == "shininess") {
					validinput = readvals(s, 1, values) ; 
					if (validinput) 
						shininess = values[0] ; 
				}

				else if (cmd == "directional" || cmd == "point") { 
					validinput = readvals(s, 6, values) ; // Position/color for lts.
					if (validinput) {

						light a = {};

						if (cmd == "directional") a.type = DIRECTIONAL;
						else if (cmd == "point") a.type = POINT;

						a.pos = vec3(values[0], values[1], values[2]);
						a.color = vec3(values[3], values[4], values[5]);

						scn->lights.push_back(a);
					}
				}

				else if (cmd == "camera") {
					validinput = readvals(s,10,values) ; // 10 values eye cen up fov
					if (validinput) {
						scn->eye = vec3(values[0], values[1], values[2]);
						scn->center = vec3(values[3], values[4], values[5]);
						scn->up = vec3(values[6], values[7], values[8]);
						scn->fovy = values[9];
					}
				}

				else if (cmd == "attenuation") {
					validinput = readvals(s, 3, values) ;
					if (validinput) {
						scn->attenuation = vec3(values[0], values[1], values[2]);
					}
				}

				else if (cmd == "background") {
					validinput = readvals(s, 3, values) ;
					if (validinput) {
						scn->background = vec3(values[0], values[1], values[2]);
					}
				}

				else if (cmd == "size") {
					validinput = readintvals(s, 2, intvalues) ;
					if (validinput) {
						scn->w = intvalues[0];
						scn->h = intvalues[1];
					}
				}

				else if (cmd == "maxdepth") {
					validinput = readintvals(s, 1, intvalues) ;
					if (validinput) {
						scn->maxdepth = intvalues[0];
					}
				}
				else if (cmd == "maxinternaldepth") {
					validinput = readintvals(s, 1, intvalues) ;
					if (validinput) {
						scn->maxinternaldepth = intvalues[0];
					}
				}

				else if (cmd == "maxthreads") {
					validinput = readintvals(s, 1, intvalues) ;
					if (validinput) {
						scn->maxthreads = intvalues[0];
					}
				}

				else if (cmd == "output") {
					s >> scn->output_filename;
					if (scn->output_filename.find(".bmp") >= 0) scn->output_filename.replace(scn->output_filename.find(".bmp"), 4, "");
					else if (scn->output_filename.find(".png") >= 0) scn->output_filename.replace(scn->output_filename.find(".png"), 4, "");
					else if (scn->output_filename.find(".ppm") >= 0) scn->output_filename.replace(scn->output_filename.find(".ppm"), 4, "");
				}

				else if (cmd == "translate") {
					validinput = readvals(s,3,values) ; 
					if (validinput) {
						mat4 translate(1.0f, 0.0f, 0.0f, values[0], 
										0.0f, 1.0f, 0.0f, values[1], 
										0.0f, 0.0f, 1.0f, values[2], 
										0.0f, 0.0f, 0.0f, 1.0f) ; 

						mat4 &T = transfstack.top().mat ; 
						T = T * glm::transpose(translate);
						mat4 &S = invtransptransfstack.top();
						S = S * glm::inverse(translate);
					}
				}

				else if (cmd == "scale") {
					validinput = readvals(s,3,values) ; 
					if (validinput) {
						mat4 scale (values[0], 0.0f, 0.0f, 0.0f, 
									0.0f, values[1], 0.0f, 0.0f, 
									0.0f, 0.0f, values[2], 0.0f, 
									0.0f, 0.0f, 0.0f, 1.0f) ; 
						if (values[0] != values[1] || values[0] != values[2])
							transfstack.top().non_uniform_scale = true;
						mat4 &T = transfstack.top().mat ; 
						T = T * scale;
						mat4 &S = invtransptransfstack.top();
						S = S * glm::inverse(scale);
					}
				}

				else if (cmd == "rotate") {
					validinput = readvals(s,4,values) ; 
					if (validinput) {
						float radians = values[3]*pi/180.0f;
						mat4 rotate = (cos(radians)*mat4(1.0f)
										+ (1-cos(radians))*mat4(values[0]*values[0], values[1]*values[0], values[2]*values[0], 0.0f,
																values[0]*values[1], values[1]*values[1], values[2]*values[1], 0.0f,
																values[0]*values[2], values[1]*values[2], values[2]*values[2], 0.0f,
																0.0f, 0.0f, 0.0f, 1.0f)
										+ sin(radians)*mat4(0.0f, -values[2], values[1], 0.0f,
															values[2], 0.0f, -values[0], 0.0f,
															-values[1], values[0], 0.0f, 0.0f,
															0.0f, 0.0f, 0.0f, 1.0f));
						// fix the lower right 1, as it was multiplied by cos, 1-cos, sin in the terms
						rotate[3][3] = 1.0f;

						mat4 &T = transfstack.top().mat ; 
						T = T * glm::transpose(rotate);
						mat4 &S = invtransptransfstack.top();
						S = S * glm::inverse(rotate);
					}
				}



				//final
				else if (cmd == "reclight")
				{
					validinput = readvals(s, 12, values) ; 
					if (validinput) {
						rec_light rec = {
							vec3(values[0], values[1], values[2]), // color
							vec3(values[3], values[4], values[5]), // corner
							vec3(values[6], values[7], values[8]), // width
							vec3(values[9], values[10], values[11])}; // height

						rec.corner = vec3(transfstack.top().mat * vec4(rec.corner, 1.0f));
						rec.width = vec3(transfstack.top().mat * vec4(rec.width, 0.0f));
						rec.height = vec3(transfstack.top().mat * vec4(rec.height, 0.0f));

						scn->rec_lights.push_back(rec);
					}
				}

				else if (cmd == "numphotons")
				{
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
						scn->num_photons = intvalues[0];
					}
				}
				else if (cmd == "photondepth")
				{
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
						scn->max_photon_depth = intvalues[0];
					}
				}
				else if (cmd == "photonradius")
				{
					validinput = readvals(s,1,values) ; 
					if (validinput) {
						scn->loc_radius = values[0];
					}
				}
				else if (cmd == "numdiffuse")
				{
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
						scn->num_of_diffuse = intvalues[0];
					}
				}

				else if (cmd == "increment") {
					validinput = readvals(s,1,values) ; 
					if (validinput) {
						scn->inc = values[0];
					}
				}

				else if (cmd == "SSAA") {
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
						if (intvalues[0] == 1 || intvalues[0] == 2 || intvalues[0] == 4 || intvalues[0] == 8 || intvalues[0] == 16) {
							scn->MSAA = 0;
							scn->SSAA = intvalues[0];
						}
						else printf("Invalid value for SSAA %d", intvalues[0]);
					}
				}

				else if (cmd == "MSAA") {
					validinput = readintvals(s,2,intvalues) ; 
					if (validinput) {
						if ((intvalues[0] == 1 || intvalues[0] == 2 || intvalues[0] == 4 || intvalues[0] == 8 || intvalues[0] == 16) &&
							(intvalues[1] == 1 || intvalues[1] == 2 || intvalues[1] == 4 || intvalues[1] == 8 || intvalues[1] == 16) && intvalues[0] > intvalues[1]) {
							scn->MSAA = intvalues[1];
							scn->SSAA = intvalues[0];
						}
						else printf("Invalid value for MSAA %d", intvalues[0]);
					}
				}

				else if (cmd == "maxverts") {
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
					}
				}

				else if (cmd == "maxvertsnorms") {
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
					}
				}

				else if (cmd == "pushTransform") {
					transfstack.push(transfstack.top()) ; 
					invtransptransfstack.push(invtransptransfstack.top()) ;
				}
				else if (cmd == "popTransform") {
					if (transfstack.size() <= 1) 
						cerr << "Stack has no elements.  Cannot Pop\n" ; 
					else transfstack.pop() ; 

					if (invtransptransfstack.size() <= 1) 
						cerr << "Stack has no elements.  Cannot Pop\n" ; 
					else invtransptransfstack.pop() ; 
				}

				else {
					cerr << "Unknown Command: " << cmd << " Skipping \n" ; 
				}
			}
			getline (in, str) ; 
		}
	}
	else {
		cerr << "Unable to Open Input Data File " << filename << "\n" ; 
		throw 2 ; 
	}
	printf("Successfully read file\n");
}


// only works for triangles
void sse_smooth(sse_scene* scn, vector<vector<int>>& smooth) {

	for (unsigned int i = 0; i < smooth.size(); ++i) {

		__m128 norm = _mm_setzero_ps();

		for (unsigned int j = 0; j < smooth[i].size(); j+=2) {

			if (smooth[i][j+1] == 0) {
				norm = _mm_add_ps(norm, scn->triangle_properties[smooth[i][j]].norm0);
			} else if (smooth[i][j+1] == 1) {
				norm = _mm_add_ps(norm, scn->triangle_properties[smooth[i][j]].norm1);
			} else if (smooth[i][j+1] == 2) {
				norm = _mm_add_ps(norm, scn->triangle_properties[smooth[i][j]].norm2);
			}
		}

		norm = __m128_NORM3(norm);

		for (unsigned int j = 0; j < smooth[i].size(); j+=2) {

			if (smooth[i][j+1] == 0) {
				scn->triangle_properties[smooth[i][j]].norm0 = norm;
			} else if (smooth[i][j+1] == 1) {
				scn->triangle_properties[smooth[i][j]].norm1 = norm;
			} else if (smooth[i][j+1] == 2) {
				scn->triangle_properties[smooth[i][j]].norm2 = norm;
			}
		}
	}
}

void make_sserec_light(sserec_light* rec, float loc_radius) {

	rec->pts = aligned_array<__m128>(16);

	float W;
	__m128_LEN3(rec->width, W);
	unsigned int w_size = (unsigned int)ceilf(W/loc_radius);

	float H;
	__m128_LEN3(rec->height, H);
	unsigned int h_size = (unsigned int)ceilf(H/loc_radius);

	unsigned int w;
	unsigned int h;
	__m128 pt;

	for (w = 0; w <= w_size; ++w) {

		for (h = 0; h <= h_size; ++h) {

			pt = _mm_add_ps(rec->corner, _mm_add_ps(__m128_MUL_float_set(rec->width, w/(float)w_size), __m128_MUL_float_set(rec->height, h/(float)h_size)));

			rec->pts.push_back(pt);
		}
	}
}

void norm_ssetriangle(ssetriangle* tri, ssetriangle_property* triprop) {

	__m128 norm = __m128_CROSS(tri->T1, tri->T2);
	norm = __m128_NORM3(norm);

	triprop->norm0 = norm;
	triprop->norm1 = norm;
	triprop->norm2 = norm;
}

void perp_ssetriangle(ssetriangle* tri, ssetriangle_property* triprop) {

	__m128_LEN3(__m128_CROSS(__m128_NORM3(tri->T1), __m128_NORM3(tri->T2)), triprop->ang_cor);
	triprop->ang_cor = 1.0f / triprop->ang_cor;

	__m128 norm = __m128_NORM3(__m128_CROSS(tri->T1, tri->T2));
	triprop->norm = norm;
	triprop->perp0 = __m128_NORM3(__m128_CROSS(norm, _mm_sub_ps(tri->T1, tri->T2)));
	triprop->perp1 = __m128_NORM3(__m128_CROSS(norm, tri->T2));
	triprop->perp2 = __m128_NORM3(__m128_CROSS(tri->T1, norm));
}

void make_ssetriangle_photons(ssetriangle* tri, ssetriangle_property* triprop, float loc_radius) {

	unsigned int i, k, m, n;
	__declspec(align(16)) float tmp[4];
	ssephoton pht = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

	__m128_LEN3(__m128_CROSS(__m128_NORM3(tri->T1), __m128_NORM3(tri->T2)), triprop->ang_cor);
	triprop->ang_cor = 1.0f / triprop->ang_cor;

	float U;
	__m128_LEN3(tri->T1, U);
	triprop->T1_lenOloc = U/(loc_radius*triprop->ang_cor);

	unsigned int row = (unsigned int)floorf(triprop->T1_lenOloc);

	triprop->u_max = row+3;

	// prepare the grid with enough rows
	triprop->photons = (ssephoton***)malloc((triprop->u_max)*sizeof(ssephoton**));
	if (triprop->photons == NULL) {
		printf("Error, malloc failed");
		exit(1);
	}

	triprop->v_max = (unsigned int*)malloc((triprop->u_max)*sizeof(unsigned int));
	if (triprop->v_max == NULL) {
		printf("Error, malloc failed");
		exit(1);
	}
	/*
	triprop->locks = (omp_lock_t**)malloc((triprop->u_max)*sizeof(omp_lock_t*));
	if (triprop->locks == NULL) {
		printf("Error, malloc failed");
		exit(1);
	}
	*/

	float V;
	__m128_LEN3(tri->T2, V);
	triprop->T2_lenOloc = V/(loc_radius*triprop->ang_cor);

	unsigned int col = (unsigned int)floorf(triprop->T2_lenOloc);

	triprop->v_max[0] = col+3;

	// add the first padding column and the first normal column
	triprop->photons[0] = (ssephoton**)malloc(triprop->v_max[0]*sizeof(ssephoton*));
	if (triprop->photons[0] == NULL) {
		printf("Error, malloc failed");
		exit(1);
	}
	/*
	triprop->locks[0] = (omp_lock_t*)malloc((triprop->v_max[0])*sizeof(omp_lock_t));
	if (triprop->locks[0] == NULL) {
		printf("Error, malloc failed");
		exit(1);
	}
	*/

	for (k = 0; k < triprop->v_max[0]; ++k) {

		triprop->photons[0][k] = (ssephoton*)malloc(TILE*TILE*sizeof(ssephoton));

		//omp_init_lock(&triprop->locks[0][k]);

		for (m = 0; m < TILE; ++m) {

			float u = ( -1.0f + ((float)m + 0.5f)/TILEf ) / triprop->T1_lenOloc;

			for (n = 0; n < TILE; ++n) {

				float v = ((float)k - 1.0f + ((float)n + 0.5f)/TILEf)/(triprop->T2_lenOloc);

				_mm_store_ps(tmp, _mm_add_ps(tri->pt0, _mm_add_ps(__m128_MUL_float_set(tri->T1, u), __m128_MUL_float_set(tri->T2, v))));

				pht.color_noitisop[3] = tmp[1];
				pht.color_noitisop[4] = tmp[2];
				pht.color_noitisop[5] = tmp[3];

				triprop->photons[0][k][m*TILE + n] = pht;
			}
		}
	}

	triprop->v_max[1] = col+3;

	triprop->photons[1] = (ssephoton**)malloc(triprop->v_max[1]*sizeof(ssephoton*));
	if (triprop->photons[1] == NULL) {
		printf("Error, malloc failed");
		exit(1);
	}
	/*
	triprop->locks[1] = (omp_lock_t*)malloc((triprop->v_max[1])*sizeof(omp_lock_t));
	if (triprop->locks[1] == NULL) {
		printf("Error, malloc failed");
		exit(1);
	}
	*/

	for (k = 0; k < triprop->v_max[1]; ++k) {

		triprop->photons[1][k] = (ssephoton*)malloc(TILE*TILE*sizeof(ssephoton));

		//omp_init_lock(&triprop->locks[1][k]);

		for (m = 0; m < TILE; ++m) {

			float u = (((float)m + 0.5f)/TILEf ) / triprop->T1_lenOloc;

			for (n = 0; n < TILE; ++n) {

				float v = ((float)k - 1.0f + ((float)n + 0.5f)/TILEf)/(triprop->T2_lenOloc);

				_mm_store_ps(tmp, _mm_add_ps(tri->pt0, _mm_add_ps(__m128_MUL_float_set(tri->T1, u), __m128_MUL_float_set(tri->T2, v))));

				pht.color_noitisop[3] = tmp[1];
				pht.color_noitisop[4] = tmp[2];
				pht.color_noitisop[5] = tmp[3];

				triprop->photons[1][k][m*TILE + n] = pht;
			}
		}
	}

	// add the remaining row, each is based off the previous column
	for (i = 2; i < triprop->u_max; ++i) {

		triprop->v_max[i] = col+3;

		triprop->photons[i] = (ssephoton**)malloc((triprop->v_max[i])*sizeof(ssephoton*));
		if (triprop->photons[i] == NULL) {
			printf("Error, malloc failed");
			exit(1);
		}
		/*
		triprop->locks[i] = (omp_lock_t*)malloc((triprop->v_max[i])*sizeof(omp_lock_t));
		if (triprop->locks[i] == NULL) {
			printf("Error, malloc failed");
			exit(1);
		}
		*/

		for (k = 0; k < triprop->v_max[i]; ++k) {

			triprop->photons[i][k] = (ssephoton*)malloc(TILE*TILE*sizeof(ssephoton));

			//omp_init_lock(&triprop->locks[i][k]);

			for (m = 0; m < TILE; ++m) {

				float u = ( (float)i - 1.0f + ((float)m + 0.5f)/TILEf ) / triprop->T1_lenOloc;

				for (n = 0; n < TILE; ++n) {

					float v = ( (float)k - 1.0f + ((float)n + 0.5f)/TILEf ) / triprop->T2_lenOloc;

					_mm_store_ps(tmp, _mm_add_ps(tri->pt0, _mm_add_ps(__m128_MUL_float_set(tri->T1, u), __m128_MUL_float_set(tri->T2, v))));

					pht.color_noitisop[3] = tmp[1];
					pht.color_noitisop[4] = tmp[2];
					pht.color_noitisop[5] = tmp[3];

					triprop->photons[i][k][m*TILE + n] = pht;
				}
			}
		}

		// set col for this box's lower corner
		col = (unsigned int)floorf( (1.0f - ((float)(i-1))/triprop->T1_lenOloc) * triprop->T2_lenOloc );

	}
}

void norm_sseparallelogram(sseparallelogram* par, sseparallelogram_property* parprop) {

	__m128 norm = __m128_CROSS(par->T1, par->T2);
	norm = __m128_NORM3(norm);

	parprop->norm0 = norm;
	parprop->norm1 = norm;
	parprop->norm2 = norm;
	parprop->norm3 = norm;
}

void perp_sseparallelogram(sseparallelogram* par, sseparallelogram_property* parprop) {

	__m128 norm = __m128_NORM3(__m128_CROSS(par->T1, par->T2));
	parprop->norm = norm;
	parprop->perp1 = __m128_NORM3(__m128_CROSS(par->T1, norm));
	parprop->perp2 = __m128_NORM3(__m128_CROSS(norm, par->T2));
}

void make_sseparallelogram_photons(sseparallelogram* par, sseparallelogram_property* parprop, float loc_radius) {

	unsigned int i, k, m, n;
	__declspec(align(16)) float tmp[4];
	ssephoton pht = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

	__m128_LEN3(__m128_CROSS(__m128_NORM3(par->T1), __m128_NORM3(par->T2)), parprop->ang_cor);
	parprop->ang_cor = 1.0f / parprop->ang_cor;

	float U;
	__m128_LEN3(par->T1, U);
	parprop->T1_lenOloc = U/(loc_radius*parprop->ang_cor);

	unsigned int row = (unsigned int)floorf(parprop->T1_lenOloc);

	parprop->u_max = row+3;

	// prepare the grid with enough rows
	parprop->photons = (ssephoton***)malloc((parprop->u_max)*sizeof(ssephoton**));
	if (parprop->photons == NULL) {
		printf("Error, malloc failed");
		exit(1);
	}

	/*
	parprop->locks = (omp_lock_t**)malloc((parprop->u_max)*sizeof(omp_lock_t*));
	if (parprop->locks == NULL) {
		printf("Error, malloc failed");
		exit(1);
	}
	*/

	float V;
	__m128_LEN3(par->T2, V);
	parprop->T2_lenOloc = V/(loc_radius*parprop->ang_cor);

	unsigned int col = (unsigned int)floorf(parprop->T2_lenOloc);

	parprop->v_max = col+3;

	// each column has the same number of boxes
	for (i = 0; i < row + 3; ++i) {

		parprop->photons[i] = (ssephoton**)malloc((parprop->v_max)*sizeof(ssephoton*));
		if (parprop->photons[i] == NULL) {
			printf("Error, malloc failed");
			exit(1);
		}
		/*
		parprop->locks[i] = (omp_lock_t*)malloc((parprop->v_max)*sizeof(omp_lock_t));
		if (parprop->locks[i] == NULL) {
			printf("Error, malloc failed");
			exit(1);
		}
		*/

		for (k = 0; k < parprop->v_max; ++k) {

			parprop->photons[i][k] = (ssephoton*)malloc(TILE*TILE*sizeof(ssephoton));

			//omp_init_lock(&parprop->locks[i][k]);

			for (m = 0; m < TILE; ++m) {

				float u = ( (float)i - 1.0f + ((float)m + 0.5f)/TILEf ) / parprop->T1_lenOloc;

				for (n = 0; n < TILE; ++n) {

					float v = ( (float)k - 1.0f + ((float)n + 0.5f)/TILEf ) / parprop->T2_lenOloc;

					_mm_store_ps(tmp, _mm_add_ps(par->pt0, _mm_add_ps(__m128_MUL_float_set(par->T1, u), __m128_MUL_float_set(par->T2, v))));

					pht.color_noitisop[3] = tmp[1];
					pht.color_noitisop[4] = tmp[2];
					pht.color_noitisop[5] = tmp[3];

					parprop->photons[i][k][m*TILE + n] = pht;
				}
			}
		}
	}
}

void sse_readfile(const char * filename, sse_scene* scn) {
	string str, cmd ; 
	ifstream in ;
	in.open(filename) ; 
	if (in.is_open()) {

		printf("Reading File\n");

		// matrix stack to store transforms.
	
		aligned_array<fmat4_ex> transf (16);
		fmat4_ex mat1 = {make_fmat4(1.0f), false};
		transf.push_back(mat1);

		aligned_array<fmat4> invtransptransf (16);
		invtransptransf.push_back(mat1.mat);

		aligned_array<__m128> verts (16);

		vector<vector<int>> smooth;
		bool do_smooth = false;
		
		aligned_array<__m128> vertnorms (16);
		aligned_array<__m128> vertnorm_norms (16);

		__m128 ambient = _mm_set_ps(0.2f, 0.2f, 0.2f, 1.0f);
		__m128 diffuse = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
		__m128 specular = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
		__m128 emission = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
		//float shininess = 1.0f; // shininess hidden in the alpha term of specular
		//float alpha = 1.0f; // alpha is hidden within the alpha term of diffuse
		//float index = 1.0f; // index is hidden in the alpha term for emission

		getline (in, str) ; 
		while (in) {
			if ((str.find_first_not_of(" \t\r\n") != string::npos) && (str[0] != '#') && !(str[str.find_first_not_of(" \t\r\n")] == '/' && str[str.find_first_not_of(" \t\r\n")+1] == '/')) {
				// Ruled out comment and blank lines 

				stringstream s(str) ;
				s >> cmd ; 

				float values[16] ; 
				int intvalues[16] ;  
				bool validinput = false ;

				if (cmd == "vertex") {
					validinput = readvals(s, 3, values) ; 
					if (validinput) {
						verts.push_back(_mm_set_ps(values[0], values[1], values[2], 1.0f));
						smooth.push_back(vector<int>());
					}			
				}

				else if (cmd == "vertexnormal") {
					validinput = readvals(s, 6, values) ; 
					if (validinput) {
						vertnorms.push_back(_mm_set_ps(values[0], values[1], values[2], 1.0f));
						vertnorm_norms.push_back(_mm_set_ps(values[3], values[4], values[5], 0.0f));
					}
				}

				else if (cmd == "tri") {
					validinput = readintvals(s, 3, intvalues) ; 
					if (validinput) {

						ssetriangle tri = {};
						ssetriangle_property triprop = {};

						// transform points into world coordinates
						__m128 pt0 = verts[intvalues[0]];
						pt0 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt0);
						tri.pt0 = pt0;
						if (do_smooth){
							smooth[intvalues[0]].push_back(scn->triangles.size());
							smooth[intvalues[0]].push_back(0);
						}

						__m128 pt1 = verts[intvalues[1]];
						pt1 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt1);
						tri.T1 = _mm_sub_ps(pt1, pt0);
						if (do_smooth){
							smooth[intvalues[1]].push_back(scn->triangles.size());
							smooth[intvalues[1]].push_back(1);
						}

						__m128 pt2 = verts[intvalues[2]];
						pt2 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt2);
						tri.T2 = _mm_sub_ps(pt2, pt0);
						if (do_smooth){
							smooth[intvalues[2]].push_back(scn->triangles.size());
							smooth[intvalues[2]].push_back(2);
						}

						float top[4], bot[4];

						_mm_storeu_ps(top, _mm_max_ps(pt0, _mm_max_ps(pt1, pt2)));
						_mm_storeu_ps(bot, _mm_min_ps(pt0, _mm_min_ps(pt1, pt2)));

						bound_box box = {TRIANGLE, scn->triangles.size(), top[3], bot[3], top[2], bot[2], top[1], bot[1]};

						scn->boxes.push_back(box);

						// get normals in world coordinates
						norm_ssetriangle(&tri, &triprop);

						perp_ssetriangle(&tri, &triprop);

						triprop.ambient = ambient ;
						triprop.emission = emission ;
						triprop.diffuse = diffuse ; 
						triprop.specular = specular ; 

						make_ssetriangle_photons(&tri, &triprop, scn->loc_radius);

						scn->triangles.push_back(tri);
						scn->triangle_properties.push_back(triprop);
					}
				}

				else if (cmd == "trinormal") {
					validinput = readintvals(s, 6, intvalues) ; 
					if (validinput) {

						ssetriangle tri = {};
						ssetriangle_property triprop = {};

						// transform into world coordinates
						__m128 pt0 = vertnorms[intvalues[0]];
						pt0 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt0);
						tri.pt0 = pt0;


						__m128 pt1 = vertnorms[intvalues[1]];
						pt1 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt1);
						tri.T1 = _mm_sub_ps(pt1, pt0);


						__m128 pt2 = vertnorms[intvalues[2]];
						pt2 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt2);
						tri.T2 = _mm_sub_ps(pt2, pt0);

						float top[4], bot[4];

						_mm_storeu_ps(top, _mm_max_ps(pt0, _mm_max_ps(pt1, pt2)));
						_mm_storeu_ps(bot, _mm_min_ps(pt0, _mm_min_ps(pt1, pt2)));

						bound_box box = {TRIANGLE, scn->triangles.size(), top[3], bot[3], top[2], bot[2], top[1], bot[1]};

						scn->boxes.push_back(box);

						// transform normals into world coordinates with inverse transpose
						pt0 = vertnorm_norms[intvalues[3]];
						triprop.norm0 = fmat4_MUL3___m128(invtransptransf[invtransptransf.size()-1], pt0);

						pt1 = vertnorm_norms[intvalues[4]];
						triprop.norm0 = fmat4_MUL3___m128(invtransptransf[invtransptransf.size()-1], pt1);

						pt2 = vertnorm_norms[intvalues[5]];
						triprop.norm0 = fmat4_MUL3___m128(invtransptransf[invtransptransf.size()-1], pt2);

						perp_ssetriangle(&tri, &triprop);

						triprop.ambient = ambient ;
						triprop.emission = emission ;
						triprop.diffuse = diffuse ; 
						triprop.specular = specular ; 

						make_ssetriangle_photons(&tri, &triprop, scn->loc_radius);

						scn->triangles.push_back(tri);
						scn->triangle_properties.push_back(triprop);
					}
				}

				else if (cmd == "par") {
					validinput = readintvals(s, 3, intvalues) ; 
					if (validinput) {

						sseparallelogram par = {};
						sseparallelogram_property parprop = {};

						// transform points into world coordinates
						__m128 pt0 = verts[intvalues[0]];
						pt0 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt0);
						par.pt0 = pt0;
						/*
						if (do_smooth){
							smooth[intvalues[0]].push_back(scn->triangles.size());
							smooth[intvalues[0]].push_back(0);
						}*/

						__m128 pt1 = verts[intvalues[1]];
						pt1 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt1);
						par.T1 = _mm_sub_ps(pt1, pt0);
						/*
						if (do_smooth){
							smooth[intvalues[1]].push_back(scn->triangles.size());
							smooth[intvalues[1]].push_back(1);
						}*/

						__m128 pt2 = verts[intvalues[2]];
						pt2 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt2);
						par.T2 = _mm_sub_ps(pt2, pt0);
						/*
						if (do_smooth){
							smooth[intvalues[2]].push_back(scn->triangles.size());
							smooth[intvalues[2]].push_back(2);
						}*/

						__m128 pt3 = _mm_add_ps(pt0, _mm_add_ps(pt1, pt2));

						float top[4], bot[4];

						_mm_storeu_ps(top, _mm_max_ps(_mm_max_ps(pt0, pt3), _mm_max_ps(pt1, pt2)));
						_mm_storeu_ps(bot, _mm_min_ps(_mm_min_ps(pt0, pt3), _mm_min_ps(pt1, pt2)));

						bound_box box = {TRIANGLE, scn->triangles.size(), top[3], bot[3], top[2], bot[2], top[1], bot[1]};

						scn->boxes.push_back(box);

						// get normals in world coordinates
						norm_sseparallelogram(&par, &parprop);

						perp_sseparallelogram(&par, &parprop);

						parprop.ambient = ambient ;
						parprop.emission = emission ;
						parprop.diffuse = diffuse ; 
						parprop.specular = specular ; 

						make_sseparallelogram_photons(&par, &parprop, scn->loc_radius);

						scn->parallelograms.push_back(par);
						scn->parallelogram_properties.push_back(parprop);
					}
				}

				else if (cmd == "parnormal") {
					validinput = readintvals(s, 7, intvalues) ; 
					if (validinput) {

						sseparallelogram par = {};
						sseparallelogram_property parprop = {};

						// transform into world coordinates
						__m128 pt0 = vertnorms[intvalues[0]];
						pt0 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt0);
						par.pt0 = pt0;


						__m128 pt1 = vertnorms[intvalues[1]];
						pt1 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt1);
						par.T1 = _mm_sub_ps(pt1, pt0);


						__m128 pt2 = vertnorms[intvalues[2]];
						pt2 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt2);
						par.T2 = _mm_sub_ps(pt2, pt0);

						__m128 pt3 = _mm_add_ps(pt0, _mm_add_ps(pt1, pt2));

						float top[4], bot[4];

						_mm_storeu_ps(top, _mm_max_ps(_mm_max_ps(pt0, pt3), _mm_max_ps(pt1, pt2)));
						_mm_storeu_ps(bot, _mm_min_ps(_mm_min_ps(pt0, pt3), _mm_min_ps(pt1, pt2)));

						bound_box box = {TRIANGLE, scn->triangles.size(), top[3], bot[3], top[2], bot[2], top[1], bot[1]};

						scn->boxes.push_back(box);

						// transform normals into world coordinates with inverse transpose
						pt0 = vertnorm_norms[intvalues[3]];
						parprop.norm0 = fmat4_MUL3___m128(invtransptransf[invtransptransf.size()-1], pt0);

						pt1 = vertnorm_norms[intvalues[4]];
						parprop.norm0 = fmat4_MUL3___m128(invtransptransf[invtransptransf.size()-1], pt1);

						pt2 = vertnorm_norms[intvalues[5]];
						parprop.norm0 = fmat4_MUL3___m128(invtransptransf[invtransptransf.size()-1], pt2);

						pt3 = vertnorm_norms[intvalues[6]];
						parprop.norm0 = fmat4_MUL3___m128(invtransptransf[invtransptransf.size()-1], pt3);

						perp_sseparallelogram(&par, &parprop);

						parprop.ambient = ambient ;
						parprop.emission = emission ;
						parprop.diffuse = diffuse ; 
						parprop.specular = specular ; 

						make_sseparallelogram_photons(&par, &parprop, scn->loc_radius);

						scn->parallelograms.push_back(par);
						scn->parallelogram_properties.push_back(parprop);
					}
				}

				else if (cmd == "sphere") {
					validinput = readvals(s, 4, values) ; 
					if (validinput) {

						if (transf[transf.size()-1].non_uniform_scale) {

							ssearbitrary_sphere sph = {};
							ssesphere_property sphprop = {};

							// hide radius squared in the w component of pos, only transform to each sphere's model coordinates
							sph.pos = _mm_set_ps(values[0], values[1], values[2], values[3]*values[3]); 

							sph.inversetransform = fmat4_transp(invtransptransf[invtransptransf.size()-1]);


							__m128 pt0 = _mm_set_ps(values[0] + values[3], values[1] + values[3], values[2] + values[3], 1.0f);
							pt0 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt0);

							__m128 pt1 = _mm_set_ps(values[0] + values[3], values[1] + values[3], values[2] - values[3], 1.0f);
							pt1 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt1);

							__m128 pt2 = _mm_set_ps(values[0] + values[3], values[1] - values[3], values[2] + values[3], 1.0f);
							pt2 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt2);

							__m128 pt3 = _mm_set_ps(values[0] + values[3], values[1] - values[3], values[2] - values[3], 1.0f);
							pt3 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt3);

							__m128 pt4 = _mm_set_ps(values[0] - values[3], values[1] + values[3], values[2] + values[3], 1.0f);
							pt4 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt4);

							__m128 pt5 = _mm_set_ps(values[0] - values[3], values[1] + values[3], values[2] - values[3], 1.0f);
							pt5 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt5);

							__m128 pt6 = _mm_set_ps(values[0] - values[3], values[1] - values[3], values[2] + values[3], 1.0f);
							pt6 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt6);

							__m128 pt7 = _mm_set_ps(values[0] - values[3], values[1] - values[3], values[2] - values[3], 1.0f);
							pt7 = fmat4_MUL3___m128(transf[transf.size()-1].mat, pt7);

							float top[4], bot[4];

							_mm_storeu_ps(top, _mm_max_ps(_mm_max_ps(_mm_max_ps(pt0, pt1), _mm_max_ps(pt2, pt3)), _mm_max_ps(_mm_max_ps(pt4, pt5), _mm_max_ps(pt6, pt7))));
							_mm_storeu_ps(bot, _mm_min_ps(_mm_min_ps(_mm_max_ps(pt0, pt1), _mm_min_ps(pt2, pt3)), _mm_min_ps(_mm_max_ps(pt4, pt5), _mm_min_ps(pt6, pt7))));

							bound_box box = {ARBITRARY_SPHERE, scn->arbitrary_spheres.size(), top[3], bot[3], top[2], bot[2], top[1], bot[1]};

							scn->boxes.push_back(box);

							sphprop.ambient = ambient ;
							sphprop.emission = emission ;
							sphprop.diffuse = diffuse ; 
							sphprop.specular = specular ;

							omp_init_lock(&sphprop.lock);

							sphprop.photons = aligned_array<ssephoton>(4);

							scn->arbitrary_spheres.push_back(sph);
							scn->arbitrary_sphere_properties.push_back(sphprop);

						} else {

							ssesphere sph = {};
							ssesphere_property sphprop = {};

							// transform center
							float center[4];

							fmat4 A = transf[transf.size()-1].mat;

							__m128 tranfcenter = fmat4_MUL3___m128(A, _mm_set_ps(values[0], values[1], values[2], 1.0f));

							_mm_storeu_ps(center, tranfcenter);

							// find transformed radius
							__m128 transfradius = _mm_sub_ps(fmat4_MUL3___m128(A, _mm_set_ps(values[0], values[1], values[2] + values[3], 1.0f)), tranfcenter);

							__m128_LEN3(transfradius, center[0]);

							values[0] = center[3]; // x
							values[1] = center[2]; // y
							values[2] = center[1]; // z
							values[3] = center[0]; // radius

							// hide radius squared in the w component of pos, only transform to each sphere's model coordinates
							sph.pos = _mm_set_ps(values[0], values[1], values[2], values[3]*values[3]); 

							__m128 pt0 = _mm_set_ps(values[0] + values[3], values[1] + values[3], values[2] + values[3], 1.0f);

							__m128 pt1 = _mm_set_ps(values[0] + values[3], values[1] + values[3], values[2] - values[3], 1.0f);

							__m128 pt2 = _mm_set_ps(values[0] + values[3], values[1] - values[3], values[2] + values[3], 1.0f);

							__m128 pt3 = _mm_set_ps(values[0] + values[3], values[1] - values[3], values[2] - values[3], 1.0f);

							__m128 pt4 = _mm_set_ps(values[0] - values[3], values[1] + values[3], values[2] + values[3], 1.0f);

							__m128 pt5 = _mm_set_ps(values[0] - values[3], values[1] + values[3], values[2] - values[3], 1.0f);

							__m128 pt6 = _mm_set_ps(values[0] - values[3], values[1] - values[3], values[2] + values[3], 1.0f);

							__m128 pt7 = _mm_set_ps(values[0] - values[3], values[1] - values[3], values[2] - values[3], 1.0f);

							float top[4], bot[4];

							_mm_storeu_ps(top, _mm_max_ps(_mm_max_ps(_mm_max_ps(pt0, pt1), _mm_max_ps(pt2, pt3)), _mm_max_ps(_mm_max_ps(pt4, pt5), _mm_max_ps(pt6, pt7))));
							_mm_storeu_ps(bot, _mm_min_ps(_mm_min_ps(_mm_max_ps(pt0, pt1), _mm_min_ps(pt2, pt3)), _mm_min_ps(_mm_max_ps(pt4, pt5), _mm_min_ps(pt6, pt7))));

							bound_box box = {SPHERE, scn->spheres.size(), top[3], bot[3], top[2], bot[2], top[1], bot[1]};

							scn->boxes.push_back(box);

							sphprop.ambient = ambient;
							sphprop.emission = emission;
							sphprop.diffuse = diffuse ; 
							sphprop.specular = specular ; 

							omp_init_lock(&sphprop.lock);

							sphprop.photons = aligned_array<ssephoton>(4);

							scn->spheres.push_back(sph);
							scn->sphere_properties.push_back(sphprop);
						}
					}
				}

				else if (cmd == "index") {
					validinput = readvals(s, 1, values) ; 
					if (validinput) 
						emission = _mm_blend_ps(emission, _mm_set1_ps(values[0]), 0x1); // index is hidden in the alpha term for emission
				}
				else if (cmd == "alpha") {
					validinput = readvals(s, 1, values) ; 
					if (validinput) 
						diffuse = _mm_blend_ps(diffuse, _mm_set1_ps(values[0]), 0x1); // alpha is hidden within the alpha term of diffuse
				}
				else if (cmd == "ambient") {
					validinput = readvals(s, 3, values) ; 
					if (validinput) 
						ambient = _mm_blend_ps(_mm_set_ps(values[0], values[1], values[2], 1.0f), ambient, 0x1);
				}
				else if (cmd == "diffuse") {
					validinput = readvals(s, 3, values) ; 
					if (validinput) 
						diffuse = _mm_blend_ps(_mm_set_ps(values[0], values[1], values[2], 1.0f), diffuse, 0x1);
				}
				else if (cmd == "specular") {
					validinput = readvals(s, 3, values) ; 
					if (validinput) 
						specular = _mm_blend_ps(_mm_set_ps(values[0], values[1], values[2], 1.0f), specular, 0x1);
				}
				else if (cmd == "emission") {
					validinput = readvals(s, 3, values) ; 
					if (validinput)
						emission = _mm_blend_ps(_mm_set_ps(values[0], values[1], values[2], 1.0f), emission, 0x1);
				}
				else if (cmd == "shininess") {
					validinput = readvals(s, 1, values) ; 
					if (validinput) 
						specular = _mm_blend_ps(specular, _mm_set1_ps(values[0]), 0x1); // shininess is hidden in the alpha term of specular
				}

				else if (cmd == "directional" || cmd == "point") { 
					validinput = readvals(s, 6, values) ; // Position/color for lts.
					if (validinput) {

						sselight l = {};

						if (cmd == "directional") {
							l.pos = _mm_set_ps(values[0], values[1], values[2], 0.0f);
						}
						else if (cmd == "point") {
							l.pos = _mm_set_ps(values[0], values[1], values[2], 1.0f);
						}
						
						l.color = _mm_set_ps(values[3], values[4], values[5], 1.0f);

						scn->lights.push_back(l);
					}
				}

				else if (cmd == "camera") {
					validinput = readvals(s,10,values) ; // 10 values eye cen up fov
					if (validinput) {
						scn->eye = _mm_set_ps(values[0], values[1], values[2], 1.0f);
						scn->center = _mm_set_ps(values[3], values[4], values[5], 1.0f);
						scn->up = _mm_set_ps(values[6], values[7], values[8], 0.0f);
						scn->fovy = values[9];
					}
				}

				else if (cmd == "attenuation") {
					validinput = readvals(s, 3, values) ;
					if (validinput) {
						scn->attenuation[0] = values[0];
						scn->attenuation[1] = values[1];
						scn->attenuation[2] = values[2];
					}
				}

				else if (cmd == "background") {
					validinput = readvals(s, 3, values) ;
					if (validinput) {
						scn->background = _mm_set_ps(values[0], values[1], values[2], 1.0f);
					}
				}

				else if (cmd == "size") {
					validinput = readintvals(s, 2, intvalues) ;
					if (validinput) {
						scn->w = intvalues[0];
						scn->h = intvalues[1];
					}
				}

				else if (cmd == "maxdepth") {
					validinput = readintvals(s, 1, intvalues) ;
					if (validinput) {
						scn->maxdepth = intvalues[0];
					}
				}
				else if (cmd == "maxinternaldepth") {
					validinput = readintvals(s, 1, intvalues) ;
					if (validinput) {
						scn->maxinternaldepth = intvalues[0];
					}
				}

				else if (cmd == "kd_depth") {
					validinput = readintvals(s, 1, intvalues) ;
					if (validinput) {
						scn->max_kd_depth = intvalues[0];
					}
				}
				else if (cmd == "kd_leaf_size") {
					validinput = readintvals(s, 1, intvalues) ;
					if (validinput) {
						scn->min_kd_leaf_size = intvalues[0];
					}
				}

				else if (cmd == "maxthreads") {
					validinput = readintvals(s, 1, intvalues) ;
					if (validinput) {
						scn->maxthreads = intvalues[0];
					}
				}

				else if (cmd == "smooth") {
					do_smooth = true;
				}
				else if (cmd == "stopsmooth") {
					do_smooth = false;
				}

				else if (cmd == "output") {
					s >> scn->output_filename;
					if (scn->output_filename.find(".bmp") >= 0) scn->output_filename.replace(scn->output_filename.find(".bmp"), 4, "");
					else if (scn->output_filename.find(".png") >= 0) scn->output_filename.replace(scn->output_filename.find(".png"), 4, "");
					else if (scn->output_filename.find(".ppm") >= 0) scn->output_filename.replace(scn->output_filename.find(".ppm"), 4, "");
				}

				else if (cmd == "translate") {
					validinput = readvals(s,3,values) ; 
					if (validinput) {
						fmat4 translate = make_fmat4(1.0f, 0.0f, 0.0f, values[0], 
													 0.0f, 1.0f, 0.0f, values[1], 
													 0.0f, 0.0f, 1.0f, values[2], 
													 0.0f, 0.0f, 0.0f, 1.0f) ; 

						fmat4& T = transf[transf.size()-1].mat ; 
						T = fmat4_mul_fmat4(T, translate);

						fmat4 invtransptranslate = make_fmat4(1.0f, 0.0f, 0.0f, 0.0f, 
															  0.0f, 1.0f, 0.0f, 0.0f, 
															  0.0f, 0.0f, 1.0f, 0.0f, 
															  -values[0], -values[1], -values[2], 1.0f) ; 
						fmat4& S = invtransptransf[invtransptransf.size()-1] ;
						S = fmat4_mul_fmat4(S, invtransptranslate);
					}
				}

				else if (cmd == "scale") {
					validinput = readvals(s,3,values) ; 
					if (validinput) {
						fmat4 scale = make_fmat4(values[0], values[1], values[2], 1.0f) ; 

						if (values[0] != values[1] || values[0] != values[2]) {

							transf[transf.size()-1].non_uniform_scale = true;
						}

						fmat4& T = transf[transf.size()-1].mat ; 
						T = fmat4_mul_fmat4(T, scale);

						fmat4 invtranspscale = make_fmat4(1.0f/values[0], 1.0f/values[1], 1.0f/values[2], 1.0f) ; 
						fmat4& S = invtransptransf[invtransptransf.size()-1] ;
						S = fmat4_mul_fmat4(S, invtranspscale);
					}
				}

				else if (cmd == "rotate") {
					validinput = readvals(s,4,values) ; 
					if (validinput) {
						float radians = values[3]*pi/180.0f;
						fmat4 rotate = fmat4_add_fmat4(fmat4_add_fmat4(fmat4_mul_float(make_fmat4(1.0f), cos(radians)),
																		fmat4_mul_float(make_fmat4(values[0]*values[0], values[1]*values[0], values[2]*values[0], 0.0f,
																									values[0]*values[1], values[1]*values[1], values[2]*values[1], 0.0f,
																									values[0]*values[2], values[1]*values[2], values[2]*values[2], 0.0f,
																									0.0f, 0.0f, 0.0f, 1.0f),
																						(1-cos(radians)))),
														fmat4_mul_float(make_fmat4(0.0f, -values[2], values[1], 0.0f,
																					values[2], 0.0f, -values[0], 0.0f,
																					-values[1], values[0], 0.0f, 0.0f,
																					0.0f, 0.0f, 0.0f, 1.0f),
																		sin(radians)));
						// fix lower right 1, as was multiplied by cos, 1-cos, sin in the terms
						rotate.r4 = _mm_blend_ps(rotate.r4, _mm_set1_ps(1.0f), 0x1);

						fmat4& T = transf[transf.size()-1].mat ; 
						T = fmat4_mul_fmat4(T, rotate);
						fmat4& S = invtransptransf[invtransptransf.size()-1] ;
						S = fmat4_mul_fmat4(S, rotate);
					}
				}

				//final
				else if (cmd == "reclight")
				{
					validinput = readvals(s, 12, values) ; 
					if (validinput) {

						sserec_light rec = {};

						rec.color = _mm_set_ps(values[0], values[1], values[2], 1.0f); // color
						rec.corner = fmat4_MUL___m128(transf[transf.size()-1].mat, _mm_set_ps(values[3], values[4], values[5], 1.0f)); // corner
						rec.width = fmat4_MUL___m128(transf[transf.size()-1].mat, _mm_set_ps(values[6], values[7], values[8], 0.0f)); // width
						rec.height = fmat4_MUL___m128(transf[transf.size()-1].mat, _mm_set_ps(values[9], values[10], values[11], 0.0f)); // height

						make_sserec_light(&rec, scn->loc_radius);

						scn->rec_lights.push_back(rec);
					}
				}

				else if(cmd == "numphotons")
				{
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
						scn->num_photons = intvalues[0];
					}
				}
				else if (cmd == "photondepth")
				{
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
						scn->max_photon_depth = intvalues[0];
					}
				}
				else if (cmd == "photonradius")
				{
					validinput = readvals(s,1,values) ; 
					if (validinput) {
						scn->loc_radius = values[0];
					}
				}
				else if (cmd == "numdiffuse")
				{
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
						scn->num_of_diffuse = intvalues[0];
					}
				}

				else if (cmd == "increment") {
					validinput = readvals(s,1,values) ; 
					if (validinput) {
						scn->inc = values[0];
					}
				}

				else if (cmd == "SSAA") {
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
						if (intvalues[0] == 1 || intvalues[0] == 2 || intvalues[0] == 4 || intvalues[0] == 8 || intvalues[0] == 16) {
							scn->MSAA = 0;
							scn->SSAA = intvalues[0];
						}
						else printf("Invalid value for SSAA %d", intvalues[0]);
					}
				}

				else if (cmd == "MSAA") {
					validinput = readintvals(s,2,intvalues) ; 
					if (validinput) {
						if ((intvalues[0] == 1 || intvalues[0] == 2 || intvalues[0] == 4 || intvalues[0] == 8 || intvalues[0] == 16) &&
							(intvalues[1] == 1 || intvalues[1] == 2 || intvalues[1] == 4 || intvalues[1] == 8 || intvalues[1] == 16) && intvalues[0] > intvalues[1]) {
							scn->MSAA = intvalues[1];
							scn->SSAA = intvalues[0];
						}
						else printf("Invalid value for MSAA %d", intvalues[0]);
					}
				}

				else if (cmd == "maxverts") {
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
					}
				}

				else if (cmd == "maxvertsnorms") {
					validinput = readintvals(s,1,intvalues) ; 
					if (validinput) {
					}
				}

				else if (cmd == "pushTransform") {
					transf.push_back(transf[transf.size()-1]) ; 
					invtransptransf.push_back(invtransptransf[invtransptransf.size()-1]) ; 
					for (unsigned int i = 0; i < smooth.size(); ++i)
						smooth[i].clear();
				}

				else if (cmd == "popTransform") {
					if (transf.size() <= 1) 
						cerr << "Stack has no elements.  Cannot Pop\n" ; 
					else transf.pop();
					if (invtransptransf.size() <= 1) 
						cerr << "Stack has no elements.  Cannot Pop\n" ; 
					else invtransptransf.pop();
					if (do_smooth)
						sse_smooth(scn, smooth);
					for (unsigned int i = 0; i < smooth.size(); ++i)
						smooth[i].clear();
				}

				else {
					cerr << "Unknown Command: " << cmd << " Skipping \n" ; 
				}
			}
			getline (in, str) ; 
		}

		/*
		_aligned_free(transf);
		_aligned_free(invtransptransf);
		_aligned_free(verts);
		_aligned_free(vertnorms);
		_aligned_free(vertnorm_norms);
		*/

		transf.clear();
		invtransptransf.clear();
		verts.clear();
		vertnorms.clear();
		vertnorm_norms.clear();

		printf("Sucessfully read file\n");
	}
	else {
		cerr << "Unable to Open Input Data File " << filename << "\n" ; 
		throw 2 ; 
	}
}


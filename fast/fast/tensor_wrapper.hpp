/*
 * tensor_wrapper.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef INCLUDE_FAST_TENSOR_WRAPPER_HPP_
#define INCLUDE_FAST_TENSOR_WRAPPER_HPP_

#include <vector>
using namespace std;

namespace FAST {
	class TensorWrapper;
}

class FAST::TensorWrapper {
public:
	virtual vector<int> getShape() const { return 0; };
protected:
	float * data_handle; //TODO make gam pointer
};

#endif /* INCLUDE_FAST_TENSOR_WRAPPER_HPP_ */

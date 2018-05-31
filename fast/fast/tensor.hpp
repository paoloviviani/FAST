/*
 * tensor.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef INCLUDE_FAST_TENSOR_HPP_
#define INCLUDE_FAST_TENSOR_HPP_

#include "mxnet-cpp/MxNetCpp.h"
#include <vector>

using namespace std;

namespace FAST {

/**
 *
 */
template <typename backendTensor>
class Tensor {
public:
	Tensor(backendTensor tensor);
	Tensor(float * raw_data, vector<unsigned int> shape);
	Tensor(Tensor tensor);
	vector<unsigned int> getShape() const;
	backendTensor getFrameworkTensor() const { return backend_tensor; };
private:
	backendTensor backend_tensor;
};



}
#endif /* INCLUDE_FAST_TENSOR_HPP_ */

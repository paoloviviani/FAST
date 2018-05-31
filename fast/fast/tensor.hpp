/*
 * tensor.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef INCLUDE_FAST_TENSOR_HPP_
#define INCLUDE_FAST_TENSOR_HPP_

#include "fast/tensor_wrapper.hpp"
#include "mxnet-cpp/MxNetCpp.h"

namespace FAST {

/**
 *
 */
template <typename backendTensor>
class Tensor : TensorWrapper<Tensor> {
public:
	//TODO implement static assert for NDArray
	vector<unsigned int> getShape() const;
	backendTensor getFrameworkTensor() const { return backend_tensor; };
private:
	backendTensor backend_tensor;
};



}
#endif /* INCLUDE_FAST_TENSOR_HPP_ */

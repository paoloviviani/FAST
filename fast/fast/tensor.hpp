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
namespace mx {
	class Tensor;
}
}

class FAST::mx::Tensor : TensorWrapper {
public:
	vector<int> getShape() const { return mx_ndarray.GetShape(); };
private:
	mxnet::cpp::NDArray mx_ndarray;
};

#endif /* INCLUDE_FAST_TENSOR_HPP_ */

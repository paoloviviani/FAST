/*
 * tensor_impl.hpp
 *
 *  Created on: May 31, 2018
 *      Author: pvi
 */

#ifndef FAST_TENSOR_IMPL_HPP_
#define FAST_TENSOR_IMPL_HPP_

#include "fast/tensor.hpp"

#define tensor_type_check(condition)  static_assert( (condition), "error: incorrect or unsupported tensor type" )

/**
 * Statically check if type is supported
 */
template <typename T>
struct is_supported {
  static const bool value = false;
};

template <>
struct is_supported<mxnet::cpp::NDArray> {
  static const bool value = true;
};

/**
 * Statically check if type is MxNet NDArray
 */
template <typename T>
struct is_MxNDAarray {
  static const bool value = false;
};

template <>
struct is_MxNDAarray<mxnet::cpp::NDArray> {
  static const bool value = true;
};

/**
 * Only enters if tensor type is supported
 * @return vector of unsigned integers with the shape of the tensor
 */
vector<unsigned int> FAST::Tensor::getShape() const {
	tensor_type_check(( is_supported<backendTensor>::value  ));

	if ( is_MxNDAarray<backendTensor>::value ){
		return backend_tensor.GetShape();
	}
}

#endif /* FAST_TENSOR_IMPL_HPP_ */

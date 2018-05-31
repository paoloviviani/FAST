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

#define tensor_type_check(condition)  static_assert( (condition), "error: incorrect or unsupported tensor type" )

using namespace std;

namespace FAST {

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
 *
 */
template <typename backendTensor>
class Tensor {

	/**
	 *
	 */
	backendTensor backend_tensor;

public:

	/**
	 *
	 * @param t
	 */
	Tensor(backendTensor t) {
		tensor_type_check(( is_supported<backendTensor>::value ));
		backend_tensor = t;
	}

	/**
	 *
	 * @param raw_data
	 * @param shape
	 */
	Tensor(float * raw_data, vector<unsigned int> shape) {
		tensor_type_check(( is_supported<backendTensor>::value ));

		if ( is_MxNDAarray<backendTensor>::value )
			backend_tensor = mxnet::cpp::NDArray(raw_data, mxnet::cpp::Shape(shape), mxnet::cpp::Context::cpu());
	}
	/**
	 *
	 * @param t
	 */
	Tensor(Tensor<mxnet::cpp::NDArray> t) {
		if ( is_MxNDAarray<backendTensor>::value )
			backend_tensor = t.getFrameworkTensor();
	}

	/**
	 *
	 * @return
	 */
	vector<unsigned int> getShape() const {
		tensor_type_check(( is_supported<backendTensor>::value  ));

		if ( is_MxNDAarray<backendTensor>::value )
			return backend_tensor.GetShape();
	}

	/**
	 *
	 * @return
	 */
	backendTensor getFrameworkTensor() const { return backend_tensor; };

};

}
#endif /* INCLUDE_FAST_TENSOR_HPP_ */

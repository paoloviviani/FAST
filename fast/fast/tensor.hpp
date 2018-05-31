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
template <typename backendType>
class Tensor {

	/**
	 *
	 */
	backendType bt;

public:

	/**
	 *
	 * @param t
	 */
	Tensor(backendType t) {
		tensor_type_check(( is_supported<backendType>::value ));
		bt = t;
	}

	/**
	 * Constructor from raw data and shape
	 * @param raw_data
	 * @param shape
	 */
	Tensor(float * raw_data, vector<unsigned int> shape) {
		tensor_type_check(( is_supported<backendType>::value ));

		if ( is_MxNDAarray<backendType>::value )
			bt = mxnet::cpp::NDArray(raw_data, mxnet::cpp::Shape(shape), mxnet::cpp::Context::cpu());
	}

	/**
	 * Constructor from raw data and size only
	 * @param raw_data
	 * @param shape
	 */
	Tensor(float * raw_data, size_t size) {
		tensor_type_check(( is_supported<backendType>::value ));

		if ( is_MxNDAarray<backendType>::value )
			bt = mxnet::cpp::NDArray(raw_data, mxnet::cpp::Shape(size), mxnet::cpp::Context::cpu());
	}

	/**
	 * Copy constructor from NxNet NDArray
	 * @param t
	 */
	Tensor(Tensor<mxnet::cpp::NDArray> t) {
		if ( is_MxNDAarray<backendType>::value )
			bt = t.getFrameworkObject();
	}

	/**
	 *
	 * @return
	 */
	vector<unsigned int> getShape() const {
		tensor_type_check(( is_supported<backendType>::value  ));

		if ( is_MxNDAarray<backendType>::value )
			return bt.GetShape();
	}

	/**
	 *
	 * @return
	 */
	backendType getFrameworkObject() const { return bt; };

};

}
#endif /* INCLUDE_FAST_TENSOR_HPP_ */

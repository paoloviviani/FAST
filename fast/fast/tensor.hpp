/*
 * tensor.hpp
 *
 *  Created on: May 30, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_TENSOR_HPP_
#define FAST_FAST_TENSOR_HPP_

#include <vector>
#include "mxnet-cpp/MxNetCpp.h"

#define tensor_type_check(condition)  static_assert( (condition), "error: incorrect or unsupported tensor type" )

using namespace std;

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


namespace FAST {

/**
 * Generic tensor wrapper class
 * Encapsulates a back-end type tensor (e.g. MxNet NDArray)
 * and provides access to raw data and some facilities
 */
template <typename backendType>
class Tensor {
	/**
	 * Check if template type is supported
	 */
	tensor_type_check(( is_supported<backendType>::value ));

protected:
	/**
	 * Tensor object of deep learning framework
	 */
	float * data_;
	vector<unsigned int> shape_;

public:

	/**
	 *
	 * @param t
	 */
	Tensor();

	/**
	 * Constructor from generic framework tensor
	 * @param t
	 */
	Tensor(backendType t);

	/**
	 * Constructor from raw data and shape
	 * @param raw_data
	 * @param shape
	 */
	Tensor(const float * raw_data, vector<unsigned int> shape);

	/**
	 * Constructor from raw data and size only
	 * @param raw_data
	 * @param shape
	 */
	Tensor(const float * raw_data, size_t size);

	/**
	 * Copy constructor from NxNet NDArray
	 * @param t
	 */
	Tensor(Tensor<mxnet::cpp::NDArray> & t);

	/**
	 * Object destructor
	 * @param t
	 */
	~Tensor();

	/**
	 *
	 * @return a vector of unsigned integer with the size of each dimension of the tensor
	 */
	vector<unsigned int> getShape() const;

	/**
	 *
	 */
	void setShape(vector<unsigned int>);

	/**
	 *
	 * @return the total number of elements of the tensor
	 */
	size_t getSize() const {
		auto v = this->getShape();
		return std::accumulate(v.begin(), v.end(), 1, std::multiplies<unsigned int>());
	};

	/**
	 *
	 * @return the element at the given index, based on underlying implementation
	 */
	template<typename... Args>
	float at(Args...) const;

	/**
	 *
	 * @return a vector of unsigned integer with the size of each dimension of the tensor
	 */
	vector<float> getStdValues();

	/**
	 *
	 * @return raw pointer to tensor data
	 */
	const float * getRawPtr() {return data_;};

	/**
	 *
	 * @return the object of type defined by the deep learning framewrok
	 */
	backendType getFrameworkObject();

};

}

#endif /* FAST_FAST_TENSOR_HPP_ */

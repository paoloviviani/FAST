/*
 * mxnet_tensor.hpp
 *
 *  Created on: Jun 1, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_MXNET_TENSOR_HPP_
#define FAST_FAST_MXNET_TENSOR_HPP_

/* Included only to let eclipse resolve the names */
#include "fast/tensor.hpp"

namespace FAST {

/* Needed to prevent multiple declarations */
template class Tensor<mxnet::cpp::NDArray>;

/**
 * Empty constructor
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor() {
	data_ = NULL;
	shape_ = vector<unsigned int>();
}

/**
 * Empty constructor
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(mxnet::cpp::NDArray t) {
	data_ = new float[t.Size()];
	t.SyncCopyToCPU(data_,t.Size());
	shape_ = t.GetShape();
}

/**
 * Constructor from raw data and shape
 * @param raw_data
 * @param shape
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(const float * raw_data, vector<unsigned int> shape) : shape_(shape) {
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<unsigned int>());
	data_ = new float[size];
	std::copy(raw_data, raw_data + size, data_);
}

/**
 * Constructor from raw data and size only
 * Only for flat data
 * @param raw_data
 * @param shape
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(const float * raw_data, size_t size) {
	data_ = new float[size];
	std::copy(raw_data, raw_data + size, data_);
}

/**
 * Copy constructor from Tensor with same template argument
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(Tensor<mxnet::cpp::NDArray> & t) {
	size_t size = t.getSize();
	data_ = new float[size];
	std::copy(t.getRawPtr(), t.getRawPtr() + size, data_);
}

/**
 * Destructor
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::~Tensor() {
	delete[] data_;
}

/**
 *
 * @param h
 * @param w
 * @return
 */
template <>
inline
size_t Tensor<mxnet::cpp::NDArray>::Offset(size_t h, size_t w) const {
  return (h *shape_[1]) + w;
}

/**
 *
 * @param c
 * @param h
 * @param w
 * @return
 */
template <>
inline
size_t Tensor<mxnet::cpp::NDArray>::Offset(size_t c, size_t h, size_t w) const {
  return h * shape_[0] * shape_[2] + w * shape_[0] + c;
}

/**
 *
 * @param h
 * @param w
 * @return
 */
template <>
inline
float Tensor<mxnet::cpp::NDArray>::At(size_t h, size_t w) const {
  return data_[Offset(h, w)];
}

/**
 *
 * @param c
 * @param h
 * @param w
 * @return
 */
template <>
inline
float Tensor<mxnet::cpp::NDArray>::At(size_t c, size_t h, size_t w) const {
  return data_[Offset(c, h, w)];
}

/**
 *
 * @return the value at the specified position
 */
template <>
template <typename... Args>
float Tensor<mxnet::cpp::NDArray>::at(Args... args) const {
	return this->At(std::forward<Args>(args)...);
}

/**
 *
 * @return raw pointer of aligned tensor data for NDArray
 */
template <>
const float * Tensor<mxnet::cpp::NDArray>::getRawPtr() {
	return data_;
}

/**
 *
 * @return a vector with each dimension of the tensor
 */
template <>
vector<unsigned int> Tensor<mxnet::cpp::NDArray>::getShape() const {
	return shape_;
}

/**
 *
 * @return a vector with each dimension of the tensor
 */
template <>
void Tensor<mxnet::cpp::NDArray>::setShape(vector<unsigned int> shape) {
	shape_ = shape;
}

/**
 *
 * @return a vector with flattened tensor values
 */
template <>
vector<float> Tensor<mxnet::cpp::NDArray>::getStdValues() {
	vector<float> out(data_,data_+this->getSize());
	return out;
}

/**
 *
 * @return
 */
template <>
mxnet::cpp::NDArray Tensor<mxnet::cpp::NDArray>::getFrameworkObject() {
	return mxnet::cpp::NDArray(data_,shape_,mxnet::cpp::Context::cpu());
}

}


#endif /* FAST_FAST_MXNET_TENSOR_HPP_ */

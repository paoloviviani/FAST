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

/* Auxiliary derived class, used for accessing elements with MxNet specific functions not to be shared with base class*/
class MxNetTensor : public Tensor<mxnet::cpp::NDArray> {
public:
	/**
	 * return offset of the element at (h, w)
	 * \param h height position
	 * \param w width position
	 * \return offset of two dimensions array
	 */
	inline
	size_t Offset(size_t h = 0, size_t w = 0) const {
		return (h *shape_[1]) + w;
	}
	/**
	 * return offset of three dimensions array
	 * \param c channel position
	 * \param h height position
	 * \param w width position
	 * \return offset of three dimensions array
	 */
	inline
	size_t Offset(size_t c, size_t h, size_t w) const {
		return h * shape_[0] * shape_[2] + w * shape_[0] + c;
	}
	/**
	 * return value of the element at (i)
	 * \param i position
	 * \return value of one dimensions array
	 */
	inline
	float At(size_t i) const {
		return data_.get()[i];
	}
	/**
	 * return value of the element at (h, w)
	 * \param h height position
	 * \param w width position
	 * \return value of two dimensions array
	 */
	inline
	float At(size_t h, size_t w) const {
		return data_.get()[Offset(h, w)];
	}
	/**
	 * return value of three dimensions array
	 * \param c channel position
	 * \param h height position
	 * \param w width position
	 * \return value of three dimensions array
	 */
	inline
	float At(size_t c, size_t h, size_t w) const {
		return data_.get()[Offset(c, h, w)];
	}
};

/* Tensor class partial template specialization implementations */

/**
 * Empty constructor
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor() {
	data_ = std::unique_ptr<float>();
	shape_ = vector<unsigned int>();
}

/**
 * Empty constructor
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(mxnet::cpp::NDArray t) {
	data_ = std::unique_ptr<float>(new float[t.Size()]);
	t.SyncCopyToCPU(data_.get(),t.Size());
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
	data_ = std::unique_ptr<float>(new float[size]);
	std::copy(raw_data, raw_data + size, data_.get());
}

/**
 * Constructor from raw data and size only
 * Only for flat data
 * @param raw_data
 * @param shape
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(const float * raw_data, size_t size) {
	data_ = std::unique_ptr<float>(new float[size]);
	std::copy(raw_data, raw_data + size, data_.get());
}

/**
 * Copy constructor from Tensor with same template argument
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(Tensor<mxnet::cpp::NDArray> & t) {
	size_t size = t.getSize();
	data_ = std::unique_ptr<float>(new float[size]);
	std::copy(t.getRawPtr(), t.getRawPtr() + size, data_.get());
}

/**
 * Destructor
 * @param t
 */
//template <>
//Tensor<mxnet::cpp::NDArray>::~Tensor() {
//	delete[] data_;
//}

/**
 *
 * @return the value at the specified position
 */
template <>
template <typename... Args>
float Tensor<mxnet::cpp::NDArray>::at(Args... args) const {
	return static_cast<const MxNetTensor*>(this)->At(std::forward<Args>(args)...);
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
	vector<float> out(data_.get(),data_.get()+this->getSize());
	return out;
}

/**
 *
 * @return
 */
template <>
mxnet::cpp::NDArray Tensor<mxnet::cpp::NDArray>::getFrameworkObject() {
	return mxnet::cpp::NDArray(data_.get(),mxnet::cpp::Shape(shape_),mxnet::cpp::Context::cpu());
}

} // End FAST namespace


#endif /* FAST_FAST_MXNET_TENSOR_HPP_ */

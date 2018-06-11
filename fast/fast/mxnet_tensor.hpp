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
class AuxMxNetTensor : public Tensor<mxnet::cpp::NDArray> {
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
	float At(size_t i) {
		return data_.local().get()[i];
	}
	/**
	 * return value of the element at (h, w)
	 * \param h height position
	 * \param w width position
	 * \return value of two dimensions array
	 */
	inline
	float At(size_t h, size_t w) {
		return data_.local().get()[Offset(h, w)];
	}
	/**
	 * return value of three dimensions array
	 * \param c channel position
	 * \param h height position
	 * \param w width position
	 * \return value of three dimensions array
	 */
	inline
	float At(size_t c, size_t h, size_t w) {
		return data_.local().get()[Offset(c, h, w)];
	}
};

/**
 * Empty constructor
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(mxnet::cpp::NDArray t) {
	size_t size = t.Size();
	data_ = gam::make_private<float>(new float[size]);
	t.SyncCopyToCPU(data_.local().get(),t.Size());
	shape_ = t.GetShape();
}

/**
 * Copy constructor from Tensor with same template argument
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(Tensor<mxnet::cpp::NDArray> & t) {
	size_t size = t.getSize();
	data_ = gam::make_private<float>(new float[size]);
	std::copy(t.getRawPtr(), t.getRawPtr() + size, data_.local().get());
}

/**
 *
 * @return the value at the specified position
 */
template <>
template <typename... Args>
float Tensor<mxnet::cpp::NDArray>::at(Args... args) {
	return static_cast<AuxMxNetTensor*>(this)->At(std::forward<Args>(args)...);
}

/**
 *
 * @return
 */
template <>
mxnet::cpp::NDArray Tensor<mxnet::cpp::NDArray>::getFrameworkObject() {
	return mxnet::cpp::NDArray(data_.local().get(),mxnet::cpp::Shape(shape_),mxnet::cpp::Context::cpu());
}

} // End FAST namespace


#endif /* FAST_FAST_MXNET_TENSOR_HPP_ */

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
		auto data_local = data_.local();
		float out = data_local->at(i);
		data_ = gam::private_ptr<gam_vector<float>>(std::move(data_local));
		return out;
	}
	/**
	 * return value of the element at (h, w)
	 * \param h height position
	 * \param w width position
	 * \return value of two dimensions array
	 */
	inline
	float At(size_t h, size_t w) {
		auto data_local = data_.local();
		float out = data_local->at(Offset(h, w));
		data_ = gam::private_ptr<gam_vector<float>>(std::move(data_local));
		return out;
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
		auto data_local = data_.local();
		float out = data_local->at(Offset(c, h, w));
		data_ = gam::private_ptr<gam_vector<float>>(std::move(data_local));
		return out;
	}
};

/**
 * Empty constructor
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(mxnet::cpp::NDArray t) {
	size_t size = t.Size();
	data_ = gam::make_private<gam_vector<float>>();
	assert(data_ != nullptr);
	auto data_local = data_.local();
	FAST_DEBUG("Created vector with shape: " << t.GetShape());
	data_local->resize(t.Size());
	t.SyncCopyToCPU(data_local->data(),t.Size());
	data_ = gam::private_ptr<gam_vector<float>>(std::move(data_local));
	shape_ = t.GetShape();
}

/**
 * Copy constructor from Tensor with same template argument
 * @param t
 */
template <>
Tensor<mxnet::cpp::NDArray>::Tensor(Tensor<mxnet::cpp::NDArray> & t) {
	size_t size = t.getSize();
	data_ = gam::make_private<gam_vector<float>>();
	assert(data_ != nullptr);
	auto data_local = data_.local();
	FAST_DEBUG("Created vector with shape: " << t.getShape());
	data_local->resize(t.getSize());
	data_local->assign(data_local->data(),data_local->data() + data_local->size());
	data_ = gam::private_ptr<gam_vector<float>>(std::move(data_local));
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
	return mxnet::cpp::NDArray(*data_.local(),mxnet::cpp::Shape(shape_),mxnet::cpp::Context::cpu());
}

} // End FAST namespace


#endif /* FAST_FAST_MXNET_TENSOR_HPP_ */

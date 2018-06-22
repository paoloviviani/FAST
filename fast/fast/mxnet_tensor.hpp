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

template <typename T>
Tensor<T>::Tensor(mxnet::cpp::NDArray & t) {
	size_t size = t.Size();
	data_ = gam::make_private<gam_vector<T>>();
	assert(data_ != nullptr);
	auto data_local = data_.local();
	FAST_DEBUG("Created vector with shape: " << t.GetShape());
	data_local->resize(t.Size());
	t.SyncCopyToCPU(data_local->data(),t.Size());
	data_ = gam::private_ptr<gam_vector<T>>(std::move(data_local));
	shape_ = t.GetShape();
}


} // End FAST namespace


#endif /* FAST_FAST_MXNET_TENSOR_HPP_ */

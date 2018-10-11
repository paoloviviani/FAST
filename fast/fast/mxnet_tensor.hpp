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
	size_ = t.Size();
	data_ = gam::make_public<gam_vector<T>>();
	assert(data_ != nullptr);
	auto data_local = data_.local();
	data_local->resize(t.Size());
	t.SyncCopyToCPU(data_local->data(),t.Size());
	data_ = gam::public_ptr<gam_vector<T>>(std::move(data_local));
}

template <typename T>
void Tensor<T>::append(mxnet::cpp::NDArray & t){
	size_t append_size = t.Size();
	auto data_local = data_.local();
	data_local->resize(size_ + append_size);
	t.SyncCopyToCPU(data_local->data() + size_,append_size);
	data_ = gam::public_ptr<gam_vector<T>>(std::move(data_local));
	size_ += append_size;
}

} // End FAST namespace


#endif /* FAST_FAST_MXNET_TENSOR_HPP_ */

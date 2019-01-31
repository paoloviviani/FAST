/*
 * Copyright (c) 2019 alpha group, CS department, University of Torino.
 *
 * This file is part of gam
 * (see https://github.com/alpha-unito/gam).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @brief       a short description of the source file
 *
 */
#ifndef INCLUDE_GAM_BACKEND_PTR_HPP_
#define INCLUDE_GAM_BACKEND_PTR_HPP_

#include "gam/defs.hpp"
#include "gam/TrackingAllocator.hpp"

namespace gam {

/*
 *
 */
class backend_ptr {
 public:
  virtual ~backend_ptr() {}

  virtual void *get() const = 0;
  virtual marshalled_t marshall() const = 0;
};

template <typename T, typename Deleter>
class backend_typed_ptr : public backend_ptr {
 public:
  backend_typed_ptr(T *ptr, Deleter d) : ptr(ptr), d(d) {}

  ~backend_typed_ptr() { d(ptr); }

  void *get() const { return (void *)ptr; }

  T *typed_get() const { return ptr; }

  marshalled_t marshall_(std::true_type) const {
    return marshalled_t(1, {ptr, sizeof(T)});
  }

  marshalled_t marshall_(std::false_type) const { return ptr->marshall(); }

  marshalled_t marshall() const {
    return marshall_(std::is_trivially_copyable<T>{});
  }

 private:
  T *ptr;
  Deleter d;
};

} /* namespace gam */

#endif /* INCLUDE_GAM_BACKEND_PTR_HPP_ */

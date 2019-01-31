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
 * @brief       implements wrapped_allocator class
 *
 * @ingroup internals
 *
 */
#ifndef INCLUDE_GAM_WRAPPED_ALLOCATOR_HPP_
#define INCLUDE_GAM_WRAPPED_ALLOCATOR_HPP_

#include <cstdlib>

#include "gam/TrackingAllocator.hpp"

namespace gam {

class wrapped_allocator {
 public:
  /*
   * shortcuts
   */
  inline void *malloc(size_t size) {
#ifdef GAM_DBG
    return a.malloc(size);
#else
    return ::malloc(size);
#endif
  }

  inline void free(void *ptr) {
#ifdef GAM_DBG
    a.free(ptr);
#else
    ::free(ptr);
#endif
  }

  template <typename obj_t, typename... Params>
  inline obj_t *new_(Params... p) {
#ifdef GAM_DBG
    obj_t *ptr = (obj_t *)this->malloc(sizeof(obj_t));
    a.new_(ptr);
    return new (ptr) obj_t(p...);
#else
    return new obj_t(p...);
#endif
  }

  template <typename T>
  inline void delete_(T *ptr) {
#ifdef GAM_DBG
    ptr->~T();
    a.delete_(ptr);
    this->free(ptr);
#else
    delete (ptr);
#endif
  }

 private:
#ifdef GAM_DBG
  TrackingAllocator a;
#endif
};

}  // namespace gam
/* namespace gam */

#endif /* GAM_INCLUDE_TRACKINGALLOCATOR_HPP_ */

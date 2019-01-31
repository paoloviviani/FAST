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
 * @brief       implements TrackingAllocator class
 *
 * @ingroup internals
 *
 */
#ifndef INCLUDE_GAM_TRACKINGALLOCATOR_HPP_
#define INCLUDE_GAM_TRACKINGALLOCATOR_HPP_

#include <mutex>
#include <unordered_map>

#include "gam/ConcurrentMapWrap.hpp"

namespace gam {

/**
 * A naive concurrent allocator with leak tracking, in form of singleton class.
 */
class TrackingAllocator {
  enum alloc_op { MALLOC_ = 0, NEW_ = 1 };
  typedef ConcurrentMapWrap<std::unordered_map<void *, alloc_op>> alloc_map_t;

 public:
  ~TrackingAllocator() {
    if (!inflight.empty()) {
      for (auto it : inflight)
        fprintf(stderr, "ALC %p %d\n", it.first, it.second);
      assert(false);
    }
  }

  void *malloc(size_t size) {
    void *res = ::malloc(size);
    assert(res);

    mtx.lock();
    assert(inflight.find(res) == inflight.end());
    mtx.unlock();

    mtx.lock();
    inflight[res] = MALLOC_;
    mtx.unlock();

    return res;
  }

  void free(void *p) {
    assert(p != nullptr);

    mtx.lock();
    assert(inflight.find(p) != inflight.end());
    mtx.unlock();

    mtx.lock();
    assert(inflight[p] == MALLOC_);
    mtx.unlock();

    mtx.lock();
    alloc_map_t::size_type res = inflight.erase(p);
    mtx.unlock();

    assert(res > 0);

    ::free(p);
  }

  void new_(void *p) {
    assert(p != nullptr);

    mtx.lock();
    assert(inflight.find(p) != inflight.end());
    mtx.unlock();

    mtx.lock();
    assert(inflight[p] == MALLOC_);
    mtx.unlock();

    mtx.lock();
    inflight[p] = NEW_;
    mtx.unlock();
  }

  void delete_(void *p) {
    assert(p != nullptr);

    mtx.lock();
    assert(inflight.find(p) != inflight.end());
    mtx.unlock();

    mtx.lock();
    assert(inflight[p] == NEW_);
    mtx.unlock();

    mtx.lock();
    inflight[p] = MALLOC_;
    mtx.unlock();
  }

 private:
  std::mutex mtx;
  alloc_map_t inflight;
};

}  // namespace gam
/* namespace gam */

#endif /* INCLUDE_GAM_TRACKINGALLOCATOR_HPP_ */

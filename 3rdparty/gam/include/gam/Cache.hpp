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
 * @date Apr 10, 2017
 * @brief implements Cache class.
 *
 * @todo bounded size
 */

#ifndef INCLUDE_GAM_CACHE_HPP_
#define INCLUDE_GAM_CACHE_HPP_

#include <unordered_map>

#include "gam/backend_ptr.hpp"
#include "gam/Logger.hpp"
#include "gam/wrapped_allocator.hpp"

namespace gam {

class Cache {
 public:
  Cache(wrapped_allocator &wa_) : wa(wa_) {}

  void finalize() {
    for (auto it : cache_map) wa.delete_(it.second);
    cache_map.clear();
  }

  template <typename T>
  void store(uint64_t a, T *p) {
    using bp_t = backend_typed_ptr<T, void (*)(T *)>;

    if (!available()) make_room();
    assert(cache_map.find(a) == cache_map.end());
    auto lp = new T();
    cache_map[a] = wa.new_<bp_t>(lp, [](T *p) { delete p; });
    *lp = *p;
    LOGLN_OS("CTX cache store a=" << a << " p=" << lp);
  }

  template <typename T>
  bool load(T *dst, uint64_t a) {
    auto it = cache_map.find(a);
    if (it != cache_map.end()) {
      LOGLN_OS("CTX cache hit a=" << a);
      *dst = *static_cast<T *>(it->second->get());
      return true;
    }
    LOGLN_OS("CTX cache miss a=" << a);
    return false;
  }

 private:
  std::unordered_map<uint64_t, backend_ptr *> cache_map;
  wrapped_allocator &wa;

  // todo fixed capacity or whatever
  bool available() { return true; }

  // todo eviction policy
  void make_room() {}
};

} /* namespace gam */

#endif /* INCLUDE_GAM_CACHE_HPP_ */

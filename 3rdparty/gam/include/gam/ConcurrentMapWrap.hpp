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

#ifndef INCLUDE_GAM_CONCURRENTMAPWRAP_HPP_
#define INCLUDE_GAM_CONCURRENTMAPWRAP_HPP_

#include <mutex>

namespace gam {

template <typename internal_map_t>
class ConcurrentMapWrap {
 public:
  typedef typename internal_map_t::key_type key_type;
  typedef typename internal_map_t::mapped_type mapped_type;
  typedef typename internal_map_t::const_iterator const_iterator;
  typedef typename internal_map_t::size_type size_type;

  bool empty() {
    mtx.lock();
    bool res = map.empty();
    mtx.unlock();

    return res;
  }

  const_iterator begin() {
    mtx.lock();
    const_iterator res = map.begin();
    mtx.unlock();

    return res;
  }

  const_iterator end() { return map.end(); }

  const_iterator find(const key_type &k) {
    mtx.lock();
    const_iterator res = map.find(k);
    mtx.unlock();

    return res;
  }

  mapped_type &operator[](const key_type &k) {
    mtx.lock();
    mapped_type &res(map[k]);
    mtx.unlock();

    return res;
  }

  size_type erase(const key_type &k) {
    mtx.lock();
    size_type res = map.erase(k);
    mtx.unlock();

    return res;
  }

 private:
  internal_map_t map;
  std::mutex mtx;
};

} /* namespace gam */

#endif /* INCLUDE_GAM_CONCURRENTMAPWRAP_HPP_ */

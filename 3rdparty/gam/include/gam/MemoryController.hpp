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
#ifndef INCLUDE_GAM_MEMORYCONTROLLER_HPP_
#define INCLUDE_GAM_MEMORYCONTROLLER_HPP_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>

#include "gam/Logger.hpp"

namespace gam {

/*
 *
 */
class MemoryController {
 public:
  inline void rc_init(uint64_t a) {
    LOGLN("SMC init %llu", a);

    // mtx.lock();
    assert(ref_cnt.find(a) == ref_cnt.end());
    // mtx.unlock();

    // mtx.lock();
    ref_cnt[a].store(1);
    // mtx.unlock();
  }

  inline unsigned long long rc_inc(uint64_t a) {
    // mtx.lock();
    unsigned long long res = ++ref_cnt[a];
    // mtx.unlock();

    LOGLN("SMC +1 %llu = %llu", a, res);
    return res;
  }

  inline unsigned long long rc_dec(uint64_t a) {
    // mtx.lock();
    unsigned long long res = --ref_cnt[a];
    // mtx.unlock();

    LOGLN("SMC -1 %llu = %llu", a, res);
    return res;
  }

  inline unsigned long long rc_get(uint64_t a) {
    unsigned long long res = ref_cnt[a];
    LOGLN("SMC %llu = %llu", a, res);
    return res;
  }

 private:
  /*
   * Reference count, for each address the process is author of.
   */
  std::unordered_map<uint64_t, std::atomic<unsigned long long>> ref_cnt;
  // std::mutex mtx;
};

} /* namespace gam */

#endif /* INCLUDE_GAM_MEMORYCONTROLLER_HPP_ */

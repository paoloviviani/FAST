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

#include <cassert>
#include <iostream>

#include "gam.hpp"

/*
 *******************************************************************************
 *
 * type definition
 *
 *******************************************************************************
 */
/*
 * The type must be DefaultConstructible and CopyAssignable.
 * Calling local() on a public pointer results in default construction followed
 * by copy assignment from (a copy of) the pointed object.
 */
template <typename T>
struct gam_indirect_vector {
  using vsize_t = typename std::vector<T>::size_type;
  vsize_t size_ = 0;

  explicit gam_indirect_vector() : vptr(nullptr) {}

  explicit gam_indirect_vector(vsize_t size, const T &v)
      : vptr(new std::vector<T>(size, v)) {}

  gam_indirect_vector &operator=(const gam_indirect_vector &copy) {
    vptr = new std::vector<T>(*copy.vptr);
    return *this;
  }

  ~gam_indirect_vector() {
    assert(vptr);
    delete vptr;
    vptr = nullptr;
  }

  /* ingesting constructor */
  template <typename StreamInF>
  void ingest(StreamInF &&f) {
    typename std::vector<T>::size_type in_size;
    f(&in_size, sizeof(vsize_t));
    assert(!this->vptr);
    this->vptr = new std::vector<T>();
    this->vptr->resize(in_size);
    assert(this->vptr->size() == in_size);
    f(this->vptr->data(), in_size * sizeof(T));
  }

  /* marshalling function */
  gam::marshalled_t marshall() {
    gam::marshalled_t res;
    size_ = this->vptr->size();
    res.emplace_back(&size_, sizeof(vsize_t));
    res.emplace_back(this->vptr->data(), size_ * sizeof(T));
    return res;
  }

  std::vector<T> *get() { return vptr; }

 private:
  std::vector<T> *vptr = nullptr;
};

template <typename T>
void populate(std::vector<T> &v) {
  for (typename std::vector<T>::size_type i = 0; i < 100; ++i) v.push_back(i);
}

/*
 *******************************************************************************
 *
 * rank-specific routines
 *
 *******************************************************************************
 */
void r0() {
  // check local consistency on public pointer
  std::shared_ptr<gam_indirect_vector<int>> lp = nullptr;
  {
    auto p = gam::make_public<gam_indirect_vector<int>>(10, 42);
    lp = p.local();
    // here end-of-scope triggers the destructor on the original object
  }
  std::vector<int> ref(10, 42);
  assert(*lp->get() == ref);

  auto p = gam::make_public<gam_indirect_vector<int>>(10, 43);
  p.push(1);

  auto q = gam::make_private<gam_indirect_vector<int>>(10, 44);
  q.push(1);
}

void r1() {
  std::shared_ptr<gam_indirect_vector<int>> lp = nullptr;
  {
    auto p = gam::pull_public<gam_indirect_vector<int>>(0);
    lp = p.local();
    // here end-of-scope triggers the destructor on the original object
  }
  std::vector<int> ref(10, 43);
  assert(*lp->get() == ref);

  auto q = gam::pull_private<gam_indirect_vector<int>>(0);
  auto lq = q.local();
  std::vector<int> qref(10, 44);
  assert(*lq->get() == qref);
}

/*
 *******************************************************************************
 *
 * main
 *
 *******************************************************************************
 */
int main(int argc, char *argv[]) {
  /* rank-specific code */
  switch (gam::rank()) {
    case 0:
      r0();
      break;
    case 1:
      r1();
      break;
  }

  return 0;
}

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
 *
 * @brief       testing compilation with multiple translation units - ranks
 *
 */

#include "gam.hpp"

#include "mtu.h"

typedef int val_t;

/*
 *******************************************************************************
 *
 * rank-specific routines
 *
 *******************************************************************************
 */
void r0() {
  /* create (with custom tracking deleter), do not access, do not push */
  gam::private_ptr<val_t> z(gam::NEW<val_t>(42), gam::DELETE<val_t>);
  assert(z != nullptr);

  /* create (with custom tracking deleter), access, do not push */
  auto p = gam::make_private<val_t>(42);
  assert(p != nullptr);
  assert(*p.local() == 42);

  /* create, push */
  auto q = gam::make_private<val_t>(42);
  assert(q != nullptr);
  q.push(1);

  /* create, access, write back, push */
  auto r = gam::make_private<val_t>(42);
  assert(r != nullptr);
  auto r_ = r.local();
  *r_ = 43;
  gam::private_ptr<val_t>(std::move(r_)).push(1);

  /* create, access, write back, push */
  auto s = gam::make_private<val_t>(42);
  assert(s != nullptr);
  auto s_ = s.local();
  (*s_)++;
  gam::private_ptr<val_t>(std::move(s_)).push(1);
}

void r1() {
  /* pull, access, do not push */
  auto p = gam::pull_private<val_t>(0);  // copy-elision (no move-construction)
  assert(p != nullptr);
  assert(*p.local() == 42);

  /* pull, access, write back, push */
  auto q = gam::pull_private<val_t>(0);
  assert(q != nullptr);
  gam::private_ptr<val_t>(q.local()).push(2);

  /* pull, access, write back, push */
  auto s = gam::pull_private<val_t>(0);
  assert(s != nullptr);
  auto s_ = s.local();
  assert(*s_ == 43);
  *s_ = 44;
  gam::private_ptr<val_t>(std::move(s_)).push(2);
}

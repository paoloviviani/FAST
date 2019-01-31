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
 * @brief       implements public_ptr class and generators
 *
 * @ingroup api
 *
 */
#ifndef INCLUDE_GAM_PUBLIC_PTR_HPP_
#define INCLUDE_GAM_PUBLIC_PTR_HPP_

#include <cstdint>
#include <unordered_map>

#include "gam/Context.hpp"  //ctx
#include "gam/GlobalPointer.hpp"
#include "gam/Logger.hpp"
#include "gam/gam_unique_ptr.hpp"
#include "gam/private_ptr.hpp"

namespace gam {

/**
 * @brief class representing public pointers.
 */
template <typename T>
class public_ptr {
 public:
  public_ptr() noexcept {}

  /**
   * @brief public pointer constructor from local pointer with custom deleter.
   *
   * It constructs a public pointer by wrapping a local pointer.
   * The deleter is used to free the local pointer.
   *
   * @param lp the local pointer to wrap
   * @param d the deleter
   */
  template <typename Deleter>
  public_ptr(T *const lp, Deleter d) : public_ptr() {
    if (lp) {
      LOGLN_OS("PUB constructor local=" << lp);
      internal_gp = ctx().mmap_public(*lp, d);
      if (internal_gp.is_address())
        ctx().rc_init(internal_gp);
      else
        std::cerr << "> could not create a public pointer for local pointer: "
                  << lp << std::endl;
    }
  }

  /**
   * @brief public pointer constructor.
   *
   * It constructs a public pointer by wrapping a global pointer.
   *
   * @param p the global pointer to wrap
   */
  public_ptr(const GlobalPointer &p) noexcept : internal_gp(p) {
    if (p.is_address() || p.address() != 0)
      LOGLN_OS("PUB constructor global=" << p);
  }

  ~public_ptr() {
    if (internal_gp.is_address()) {
      LOGLN_OS("PUB destroy global=" << internal_gp);
      reset();
    }
  }

  /*
   ***************************************************************************
   *
   * copy constructor and assignment
   *
   ***************************************************************************
   */
  public_ptr(const public_ptr &copy) noexcept : internal_gp(copy.internal_gp) {
    LOGLN_OS("PUB copy-constructor global=" << internal_gp);
    if (internal_gp.is_address()) ctx().rc_inc(internal_gp);
  }

  public_ptr &operator=(const public_ptr &copy) noexcept {
    LOGLN_OS("PUB copy-assignment obj=" << copy << " sub=" << *this);

    if (internal_gp.address() != copy.internal_gp.address()) {
      if (copy.internal_gp.is_address()) ctx().rc_inc(copy.internal_gp);
      if (internal_gp.is_address()) ctx().rc_dec(internal_gp);
      internal_gp = copy.internal_gp;
    }
    return *this;
  }

  /*
   ***************************************************************************
   *
   * move constructor and assignment
   *
   ***************************************************************************
   */
  public_ptr(public_ptr &&other) noexcept : internal_gp(other.internal_gp) {
    LOGLN_OS("PUB move-constructor global=" << internal_gp);

    /* neutralize other destruction  */
    other.internal_gp.address(0);
  }

  public_ptr &operator=(public_ptr &&other) noexcept {
    LOGLN_OS("PUB move-assignment obj=" << other << " sub=" << *this);

    /* swap to cause subject destruction */
    GlobalPointer tmp = internal_gp;
    internal_gp = other.internal_gp;
    other.internal_gp = tmp;
    return *this;
  }

  /*
   ***************************************************************************
   *
   * constructor and assignment from private pointer
   *
   ***************************************************************************
   */
  public_ptr(const private_ptr<T> &copy) = delete;
  public_ptr &operator=(const private_ptr<T> &copy) = delete;

  /**
   * @brief builds a public pointer from private pointer.
   *
   * After completion, the argument private pointer is destroyed.
   */
  explicit public_ptr(private_ptr<T> &&p) noexcept {
    GlobalPointer gp = p.get();

    if (gp.is_address()) {
      LOGLN_OS("PUB from-PVT constructor from=" << gp);

      /* remap to public address */
      internal_gp = ctx().publish<T>(gp);
      p.release();

      /* init reference counter */
      ctx().rc_init(internal_gp);
    } else {
      internal_gp = gp;
    }
  }

  /**
   * @brief assigns a public pointer.
   *
   * After completion, the argument private pointer is destroyed.
   */
  public_ptr &operator=(private_ptr<T> &&p) {
    GlobalPointer gp = p.get();

    if (gp.is_address()) {
      LOGLN_OS("PUB from-PVT assignment from=" << gp << " sub=" << *this);

      if (internal_gp.is_address()) reset();

      /* remap to public address */
      internal_gp = ctx().publish<T>(gp);
      p.release();

      /* init reference counter */
      ctx().rc_init(internal_gp);
    } else {
      internal_gp = gp;
    }

    return *this;
  }

  /*
   ***************************************************************************
   *
   * GAM primitives
   *
   ***************************************************************************
   */

  /**
   * @brief generates a local copy
   *
   * local makes a local copy of the memory pointed by the public pointer and
   * returns it as a shared pointer.
   */
  std::shared_ptr<T> local() const {
    if (internal_gp.is_address()) return ctx().local_public<T>(internal_gp);
    std::cerr << "> called local() for non-address pointer:\n"
              << internal_gp << std::endl;
    return nullptr;
  }

  /**
   * @ brief pushes the pointer to another executor
   *
   * @param to is the executor to push to
   */
  void push(executor_id to) const {
    if (to < ctx().cardinality()) {
      if (internal_gp.is_address()) {
        // pointer brings a global address
        ctx().push_public(internal_gp, to);
        ctx().rc_inc(internal_gp);
      } else
        // pointer brings a reserved value
        ctx().push_reserved(internal_gp, to);
    } else
      std::cerr << "> called push() towards invalid rank: " << to << std::endl;
  }

  /*
   ***************************************************************************
   *
   * reset and get have the same semantics as in std::shared_ptr
   *
   ***************************************************************************
   */

  unsigned long long use_count() const { return ctx().rc_get(internal_gp); }

  void reset() noexcept {
    ctx().rc_dec(internal_gp);
    internal_gp.address(0);
  }

  GlobalPointer get() const noexcept { return internal_gp; }

  /**
   * @brief pretty-prints the pointer
   */
  friend std::ostream &operator<<(std::ostream &out, const public_ptr &f) {
    return out << "[PUB global=" << f.internal_gp << "]";
  }

  /*
   ***************************************************************************
   *
   * nullptr creation, comparison and assignment
   *
   ***************************************************************************
   */
  public_ptr(std::nullptr_t) noexcept {}

  public_ptr &operator=(std::nullptr_t) noexcept {
    if (internal_gp.is_address()) {
      LOGLN_OS("PUB nullptr assignment sub=" << *this);
      reset();
    }

    return *this;
  }

  bool operator==(std::nullptr_t) const noexcept {
    return internal_gp.address() == 0;
  }

  friend bool operator==(std::nullptr_t, const public_ptr &__x) noexcept {
    return __x == nullptr;
  }

  bool operator!=(std::nullptr_t) noexcept {
    return internal_gp.address() != 0;
  }

  friend bool operator!=(std::nullptr_t, const public_ptr &__x) noexcept {
    return __x != nullptr;
  }

 private:
  GlobalPointer internal_gp;
};

template <typename _Tp, typename... _Args>
public_ptr<_Tp> make_public(_Args &&... __args) {
  return public_ptr<_Tp>(                        //
      NEW<_Tp>(std::forward<_Args>(__args)...),  //
      DELETE<_Tp>);
}

/**
 * @ brief blocking pull an incoming public pointer from another executor
 *
 * @param from is the executor to pull from
 * @retval the incoming pointer
 */
template <typename T>
public_ptr<T> pull_public(executor_id from) {
  if (from < ctx().cardinality() && from != ctx().rank())
    return public_ptr<T>(ctx().pull_public(from));
  std::cerr << "> pull_public() towards invalid rank: " << from << std::endl;
  return nullptr;
}

/**
 * @ brief blocking pull an incoming public pointer from any other executor
 *
 * @retval the incoming pointer
 */
template <typename T>
public_ptr<T> pull_public() noexcept {
  return public_ptr<T>(ctx().pull_public());
}

} /* namespace gam */

#endif /* INCLUDE_GAM_PUBLIC_PTR_HPP_ */

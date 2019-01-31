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
 * @defgroup internals Internals
 *
 * all stuff working under the hood
 */

/**
 * @brief       implements Context class
 *
 * @ingroup     internals
 *
 * @todo remote-load by RMA operations + remote memory allocation
 * @todo move local memory management to a dedicated module
 * @todo friendly error reporting
 * @todo define thread-safeness for each function
 */
#ifndef CONTEXT_HPP_
#define CONTEXT_HPP_

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>  //memcpy
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "gam/Cache.hpp"
#include "gam/GlobalPointer.hpp"
#include "gam/Logger.hpp"
#include "gam/MemoryController.hpp"
#include "gam/View.hpp"
#include "gam/backend_ptr.hpp"
#include "gam/defs.hpp"
#include "gam/links_stub.hpp"
#include "gam/wrapped_allocator.hpp"

#ifdef CONNECTION_LINKS
#include "links_implementations/fl_connection.hpp"
#else
#include "gam/links_implementations/fl_connectionless.hpp"
#endif

namespace gam {

#ifdef CONNECTION_LINKS
template <typename T>
using links_impl = fl_connection;
#else
template <typename T>
using links_impl = fl_connectionless;
#endif

template <typename T>
using Links = links_stub<links_impl<T>, T>;

/*
 * forward declarations for local memory allocation
 */

template <typename T>
inline void DELETE(T *ptr);

/**
 * Context represents the executor state.
 */
class Context {
 public:
  Context() : cache(local_allocator) {
    char *env, *tmp;

    /*
     * read rank from env
     */
    env = std::getenv("GAM_RANK");
    assert(env);
    rank_ = strtoull(env, &tmp, 10);
    assert(rank_ <= GlobalPointer::max_home);

    /*
     * initialize logger
     */
    env = std::getenv("GAM_LOG_PREFIX");
    assert(env);
    LOGGER_INIT(env, rank_);
    LOGLN("CTX rank = %llu", rank_);

    /*
     * read cardiality from env
     */
    env = std::getenv("GAM_CARDINALITY");
    assert(env);
    cardinality_ = strtoull(std::getenv("GAM_CARDINALITY"), &tmp, 10);
    LOGLN("CTX cardinality = %llu", cardinality_);
    assert(cardinality_ <= GlobalPointer::max_home + 1);

    /*
     * read node and service names from env
     */
    struct node_t {
      char *host, *svc_pap, *svc_local, *svc_remote;
    } node;
    std::vector<node_t> nodes;
    std::string env_prefix = "GAM_", env_name;
    for (unsigned long long i = 0; i < cardinality_; ++i) {
      env_name = env_prefix + "NODE_" + std::to_string(i);
      node.host = std::getenv(env_name.c_str());
      assert(node.host);
      env_name = env_prefix + "SVC_PAP_" + std::to_string(i);
      node.svc_pap = std::getenv(env_name.c_str());
      assert(node.svc_pap);
      env_name = env_prefix + "SVC_MEM_" + std::to_string(i);
      node.svc_local = std::getenv(env_name.c_str());
      assert(node.svc_local);
      env_name = env_prefix + "SVC_DMN_" + std::to_string(i);
      node.svc_remote = std::getenv(env_name.c_str());
      assert(node.svc_remote);
      nodes.push_back(node);
      LOGLN("CTX rank %llu: node=%s svc_pap=%s svc_mem=%s svc_dmn=%s",  //
            i, node.host, node.svc_pap, node.svc_local, node.svc_remote);
    }

    /*
     * initialize links
     */
    Links<pap_pointer>::init_links(node.host);

    /*
     * create links
     */
    pap_links =
        new Links<pap_pointer>(cardinality_, rank_, nodes[rank_].svc_pap);
    local_links =
        new Links<daemon_pointer>(cardinality_, rank_, nodes[rank_].svc_local);
    remote_links =
        new Links<daemon_pointer>(cardinality_, rank_, nodes[rank_].svc_remote);

    /*
     * add peers
     */
    for (unsigned long long i = 0; i < cardinality_; ++i) {
      if (i != rank_) {
        pap_links->peer(i, nodes[i].host, nodes[i].svc_pap);  // send push
        local_links->peer(i, nodes[i].host,
                          nodes[i].svc_remote);  // send rc + rload req
        remote_links->peer(i, nodes[i].host,
                           nodes[i].svc_local);  // send rload rep
      }
    }

    /*
     * init links
     */
    pap_links->init(nodes[rank_].host,
                    nodes[rank_].svc_pap);  // recv push (i.e. pull)
    remote_links->init(nodes[rank_].host,
                       nodes[rank_].svc_remote);  // recv rc + rload req
    local_links->init(nodes[rank_].host,
                      nodes[rank_].svc_local);  // recv rload rep

    /*
     * spawn daemon thread
     */
    daemon = new std::thread(Daemon(*this));
  }

  ~Context() {
    /*
     * finalize and join daemon thread
     */
    daemon_termination = true;
    daemon->join();
    delete daemon;

    /*
     * finalize cache
     */
    cache.finalize();

    /*
     * finalize links
     */
    pap_links->finalize();
    local_links->finalize();
    remote_links->finalize();

    // clean-up
    delete pap_links;
    delete local_links;
    delete remote_links;

    Links<pap_pointer>::fini_links();

    /*
     * finalize logger
     */
    LOGGER_FINALIZE(rank_);
  }

  executor_id rank() const { return rank_; }

  executor_id cardinality() const { return cardinality_; }

  /*
   ***************************************************************************
   *
   * global memory mapping
   *
   ***************************************************************************
   */
  /*
   * map a fresh global address (for public memory) to local pointer
   */
  template <class T, typename Deleter>
  GlobalPointer mmap_public(T &lp, Deleter d) {
    GlobalPointer res = mmap_global<AL_PUBLIC>(&lp, d);
    uint64_t a = res.address();

    /* initialize owner and child fields with void values */
    view.bind_owner(a, (executor_id)GlobalPointer::max_home + 1);
    view.bind_child(a, nullptr);

    return res;
  }

  /*
   * map a fresh global address (for public memory) to local pointer
   */
  template <class T, typename Deleter>
  GlobalPointer mmap_private(T &lp, Deleter d) {
    GlobalPointer res = mmap_global<AL_PRIVATE>(&lp, d);
    uint64_t a = res.address();

    /* update owner */
    view.bind_owner(a, rank_);

    /* update parenthood information */
    view.bind_parent(&lp, a);
    view.bind_child(a, &lp);

    return res;
  }

  void unmap(const GlobalPointer &p) {
    assert(p.is_address());
    LOGLN_OS("CTX unmapping p=" << p);
    uint64_t a = p.address();

    /* clean up parenthood */
    if (view.access_level(a) == AL_PRIVATE) {
      assert(view.has_child(a));
      assert(view.has_parent(view.child(a)));
      view.unbind_parent(view.child(a));
    } else
      assert(!view.has_child(a));

    munmap(p);
  }

  /*
   ***************************************************************************
   *
   * capability passing: push and pull
   *
   ***************************************************************************
   */
  inline void push_public(const GlobalPointer &p, const executor_id e) {
    assert(p.is_address());
    uint64_t a = p.address();
    assert(view.access_level(a) == AL_PUBLIC);
    LOGLN_OS("CTX push public=" << p << " to=" << e);

    pap_pointer buf;
    buf.p = p;
    buf.al = AL_PUBLIC;
    buf.author = view.author(a);
    pap_links->send(buf, e);
  }

  inline void push_private(const GlobalPointer &p, const executor_id e) {
    assert(p.is_address());
    LOGLN_OS("CTX push private=" << p << " to=" << e);
    uint64_t a = p.address();
    assert(view.access_level(a) == AL_PRIVATE);
    assert(view.owner(a) == rank_);

    /* switch ownership */
    view.bind_owner(a, e);

    pap_pointer buf;
    buf.p = p;
    buf.al = AL_PRIVATE;
    buf.author = view.author(a);
    pap_links->send(buf, e);
  }

  inline void push_reserved(const GlobalPointer &p, const executor_id e) {
    assert(!p.is_address());
    LOGLN_OS("CTX push reserved =" << p << " to=" << e);

    pap_pointer buf;
    buf.p = p;
    pap_links->send(buf, e);
  }

  /*
   * blocking pull a public address from specific executor
   */
  inline GlobalPointer pull_public(const executor_id e) {
    LOGLN_OS("CTX pull public from=" << e);

    pap_pointer buf;
    pap_links->recv(buf, e);

    /* ensure a public pointer was pulled */
    if (!buf.p.is_address() || buf.al == AL_PUBLIC) return pulled_public(buf);

    std::cerr << "> pull_public() pulled a non-public pointer: \n"
              << buf.p << std::endl;
    return GlobalPointer();
  }

  /*
   * blocking pull public address from any executor
   */
  inline GlobalPointer pull_public() {
    LOGLN_OS("CTX pull public from any");

    pap_pointer buf;
    pap_links->recv(buf);

    /* ensure a public pointer was pulled */
    if (!buf.p.is_address() || buf.al == AL_PUBLIC) return pulled_public(buf);

    std::cerr << "> pull_public() pulled a non-public pointer: \n"
              << buf.p << std::endl;
    return GlobalPointer();
  }

  /*
   * blocking pull private address from specific executor
   */
  inline GlobalPointer pull_private(const executor_id e) {
    LOGLN_OS("CTX pull private from=" << e);
    pap_pointer buf;
    pap_links->recv(buf, e);

    /* ensure a private pointer was pulled */
    if (!buf.p.is_address() || buf.al == AL_PRIVATE) return pulled_private(buf);

    std::cerr << "> pull_private() pulled a non-private pointer: \n"
              << buf.p << std::endl;
    return GlobalPointer();
  }

  /*
   * blocking pull private address from any executor
   */
  inline GlobalPointer pull_private() {
    LOGLN_OS("CTX pull private from any");
    pap_pointer buf;
    pap_links->recv(buf);

    /* ensure a private pointer was pulled */
    if (!buf.p.is_address() || buf.al == AL_PRIVATE) return pulled_private(buf);

    std::cerr << "> pull_private() pulled a non-private pointer: \n"
              << buf.p << std::endl;
    return GlobalPointer();
  }

  /*
   ***************************************************************************
   *
   * conversion to local memory pointers
   *
   ***************************************************************************
   */
  /*
   * local_public generates a local copy of public memory
   */
  template <typename T>
  inline std::shared_ptr<T> local_public(const GlobalPointer &p) {
    assert(p.is_address());

    LOGLN_OS("CTX local public " << p);
    uint64_t a = p.address();
    assert(view.access_level(a) == AL_PUBLIC);

    /* allocate local memory */
    T *lp = (T *)local_new<T>();

    /* load either locally or remotely */
    if (view.author(a) == rank_)
      local_load(lp, a);
    else if (!cache.load(lp, a)) {
      forward_load(lp, p);
      cache.store(a, lp);
    }

    /* generate a smart pointer with custom deleter to match allocation */
    return std::shared_ptr<T>((T *)lp, [](T *p_) { DELETE(p_); });
  }

  /*
   * local_private returns the pointer associated to (private) global address
   */
  template <typename T>
  T *local_private(const GlobalPointer &p) {
    assert(p.is_address());
    uint64_t a = p.address();
    LOGLN_OS("CTX local private " << p);
    assert(view.access_level(a) == AL_PRIVATE);

    executor_id auth = view.author(a);

    if (auth != rank_) {
      /* steal memory and swap authorship */
      withdraw<T>(p);
      forward_reset(p, auth);
    }

    return reinterpret_cast<T *>(view.committed(a)->get());
  }

  /*
   ***************************************************************************
   *
   * publishing private addresses
   *
   ***************************************************************************
   */
  /*
   * remap private memory to a fresh public address
   */
  template <typename T>
  GlobalPointer publish(const GlobalPointer &p) {
    GlobalPointer p_;

    LOGLN_OS("CTX publishing p=" << p);
    assert(p.is_address());
    assert(is_private(p));
    assert(am_owner(p));
    uint64_t a = p.address();
    executor_id auth = view.author(a);

    /* map to fresh global address */
    p_ = name_allocator();
    uint64_t a_ = p_.address();
    view.bind_access_level(a_, AL_PUBLIC);

    /* steal memory */
    backend_ptr *bp_;
    if (auth == rank_) {
      assert(view.has_child(a));
      assert(view.has_parent(view.child(a)));

      view.unbind_parent(view.child(a));
      bp_ = view.committed(a);
    } else {
      assert(!view.has_child(a));

      /* allocate backend memory */
      T *tmp = (T *)local_new<T>();
      using bp_t = backend_typed_ptr<T, void (*)(T *)>;
      auto tbp_ = local_new<bp_t>(tmp, DELETE<T>);

      /* remote load */
      forward_load(tbp_->typed_get(), p);
      bp_ = tbp_;

      /* notify remote author  */
      forward_reset(p, auth);
    }

    /* clear old entry */
    view.unmap(a);

    /* update fresh entry */
    view.bind_committed(a_, bp_);
    view.bind_author(a_, rank_);
    view.bind_owner(a_, (executor_id)GlobalPointer::max_home + 1);  // dummy
    view.bind_child(a_, nullptr);                                   // dummy

    return p_;
  }

  /*
   ***************************************************************************
   *
   * utility query functions
   *
   ***************************************************************************
   */
  bool is_public(const GlobalPointer &p) {
    assert(p.is_address());
    return view.access_level(p.address()) == AL_PUBLIC;
  }

  bool is_private(const GlobalPointer &p) {
    assert(p.is_address());
    return view.access_level(p.address()) == AL_PRIVATE;
  }

  bool am_owner(const GlobalPointer &p) {
    assert(p.is_address());
    assert(is_private(p));
    return view.owner(p.address()) == rank_;
  }

  bool am_author(const GlobalPointer &p) {
    assert(p.is_address());
    return view.author(p.address()) == rank_;
  }

  executor_id author(const GlobalPointer &p) {
    assert(p.is_address());
    return view.author(p.address());
  }

  template <typename T>
  bool has_parent(T *lp) {
    return view.has_parent((void *)lp);
  }

  template <typename T>
  GlobalPointer parent(T *lp) {
    return GlobalPointer(view.parent((void *)lp));
  }

  /*
   ***************************************************************************
   *
   * public pointers support
   *
   ***************************************************************************
   */
  inline void rc_init(const GlobalPointer &p) {
    assert(p.is_address());
    mc.rc_init(p.address());
  }

  inline void rc_inc(const GlobalPointer &p) {
    assert(p.is_address());
    uint64_t a = p.address();
    assert(view.access_level(a) == AL_PUBLIC);

    if (view.author(a) == rank_)
      mc.rc_inc(a);
    else
      forward_inc(p);
  }

  inline void rc_dec(const GlobalPointer &p) {
    assert(p.is_address());
    uint64_t a = p.address();
    assert(view.access_level(a) == AL_PUBLIC);

    if (view.author(a) == rank_) {
      if (mc.rc_dec(a) == 0)
        /*
         * destroy and un-bind committed memory
         */
        unmap(a);
    } else
      forward_dec(p);
  }

  inline unsigned long long rc_get(GlobalPointer gp) {
    assert(gp.is_address());
    uint64_t a = gp.address();
    return view.author(a) == rank_ ? local_rc_get(a) : forward_rc(gp);
  }

  /*
   ***************************************************************************
   *
   * private pointers support
   *
   ***************************************************************************
   */
  inline void forward_reset(const GlobalPointer &p, const executor_id to) {
    assert(p.is_address());
    LOGLN("CTX fwd -1 %llu dest=%lu", p.address(), to);
    daemon_pointer dp;
    dp.op = daemon_pointer::PVT_RESET;
    dp.from = rank_;
    dp.p = p;
    local_links->send(dp, to);
  }

  /*
   ***************************************************************************
   *
   * Tracking local memory allocation
   *
   ***************************************************************************
   */
  template <typename T>
  void local_typed_free(T *ptr) {
    local_allocator.free((void *)ptr);
  }

  template <typename obj_t, typename... Params>
  obj_t *local_new(Params... p) {
    return local_allocator.new_<obj_t>(p...);
  }

  template <typename T>
  void local_delete(T *ptr) {
    local_allocator.delete_(ptr);
  }

 private:
  executor_id rank_, cardinality_;
  std::vector<std::string> hostnames;

  View view;            // concurrent memory table
  MemoryController mc;  // concurrent reference counting table
  Cache cache;

  std::thread *daemon;
  std::atomic<char> daemon_termination;

  std::mt19937 name_allocator;

  /*
   ***************************************************************************
   *
   * links are organized in different TCP services (i.e., ports)
   *
   ***************************************************************************
   */
  struct pap_pointer {
    GlobalPointer p;
    executor_id author = 0;
    AccessLevel al;
  };

  struct daemon_pointer {
    enum { RLOAD, RC_INC, RC_DEC, RC_GET, PVT_RESET, DMN_END } op;
    size_t size;  // remote-load size
    executor_id from;
    GlobalPointer p;
  };

  /*
   * links for pushing and pulling pointers (svc A)
   */
  Links<pap_pointer> *pap_links;

  /*
   * links for:
   * - sending   reference-counting and remote-load requests (svc B)
   * - receiving remote-load responses (svc C)
   */
  Links<daemon_pointer> *local_links;

  /*
   * links for:
   * - receiving reference-counting and remote-load requests (svc B)
   * - sending   remote-load responses (svc C)
   */
  Links<daemon_pointer> *remote_links;

  /*
   * Tracking local memory allocator
   */
  wrapped_allocator local_allocator;

  /*
   ***************************************************************************
   *
   * Daemon represents the utility thread associated to the execution context
   * that handles memory requests from other executors
   *
   ***************************************************************************
   */
  class Daemon {
   public:
    Daemon(Context &ctx) : ctx(ctx), cnt(ctx.cardinality_ - 1) {}

    void operator()() {
      if (cnt) {
        ctx.remote_links->nb_recv(p);

        LOGLN_OS("DMN start serving remote requests [tid="
                 << std::this_thread::get_id() << "]");
        while (!ctx.daemon_termination) poll_iteration();
      }

      /* broadcast termination to rc-consumer links */
      LOGLN("DMN broadcast termination", p.p.address());
      daemon_pointer p_end;
      p_end.op = daemon_pointer::DMN_END;
      p_end.from = ctx.rank_;
      ctx.local_links->broadcast(p_end);

      LOGLN("DMN keep serving remote requests", p.p.address());
      while (cnt) poll_iteration();
    }

   private:
    Context &ctx;
    executor_id cnt;  // terminated partitions
    daemon_pointer p;

    void poll_iteration() {
      if (ctx.remote_links->nb_poll()) {
        /* handle the incoming request */
        uint64_t a = p.p.address();
        switch (p.op) {
          case daemon_pointer::RC_INC:
            LOGLN("DMN recv +1 %llu from %lu", a, p.from);
            assert(ctx.view.author(a) == ctx.rank_);
            ctx.mc.rc_inc(a);
            break;
          case daemon_pointer::RC_DEC:
            LOGLN("DMN recv -1 %llu from %lu", a, p.from);
            assert(ctx.view.author(a) == ctx.rank());
            if (ctx.mc.rc_dec(a) == 0) ctx.unmap(a);
            break;
          case daemon_pointer::RC_GET: {
            LOGLN("DMN recv RC_GET %llu from %lu", a, p.from);
            assert(ctx.view.author(a) == ctx.rank_);
            assert(ctx.view.committed(a) != nullptr);
            unsigned long long rc = ctx.local_rc_get(a);
            ctx.remote_links->raw_send(&rc, sizeof(unsigned long long), p.from);
          } break;
          case daemon_pointer::PVT_RESET:
            LOGLN("DMN recv PVT -1 %llu from %lu", a, p.from);
            assert(ctx.view.author(a) == ctx.rank_);
            assert(ctx.view.committed(a) != nullptr);
            ctx.unmap(a);
            break;
          case daemon_pointer::RLOAD:
            LOGLN("DMN recv RLOAD %llu from %lu", a, p.from);
            assert(ctx.view.author(a) == ctx.rank_);
            assert(ctx.view.committed(a) != nullptr);
            for (auto &me : ctx.view.committed(a)->marshall())
              ctx.remote_links->raw_send(me.base, me.size, p.from);
            break;
          case daemon_pointer::DMN_END:
            LOGLN("DMN recv RC_END from %lu", p.from);
            --cnt;
            break;
          default:
            assert(false);
            break;
        }

        /* accept another token */
        ctx.remote_links->nb_recv(p);
      }
    }
  };

  template <AccessLevel al, class T, typename Deleter>
  GlobalPointer mmap_global(T *lp, Deleter d) {
    GlobalPointer res = GlobalPointer(name_allocator(), rank_);
    uint64_t a = res.address();

    LOGLN("CTX mmap global=%llu -> local=%p", a, lp);

    /* implicit commit */
    assert(view.committed(a) == nullptr);
    using bp_t = backend_typed_ptr<T, Deleter>;
    bp_t *bp = local_new<bp_t>(lp, d);
    view.bind_committed(a, bp);

    /* update view information */
    view.bind_access_level(a, al);
    view.bind_author(a, rank_);

    return res;
  }

  void munmap(const GlobalPointer &p) {
    uint64_t a = p.address();

    backend_ptr *cm = view.committed(a);
    assert(cm != nullptr);

    /* clean up view */
    view.unmap(a);

    /* finally release committed memory */
    local_delete(cm);
  }

  GlobalPointer pulled_public(const pap_pointer &buf) {
    GlobalPointer res;

    if (buf.p.is_address()) {
      LOGLN_OS("CTX pulled public=" << buf.p);

      uint64_t a = buf.p.address();
      view.bind_access_level(a, buf.al);
      view.bind_owner(a, (executor_id)GlobalPointer::max_home + 1);
      view.bind_author(a, buf.author);
      view.bind_committed(a, nullptr);
    } else
      LOGLN_OS("CTX pulled reserved=" << buf.p);

    /* prepare output */
    res = buf.p;

    return res;
  }

  GlobalPointer pulled_private(const pap_pointer &buf) {
    GlobalPointer res;

    if (buf.p.is_address()) {
      LOGLN_OS("CTX pulled private=" << buf.p);

      uint64_t a = buf.p.address();
      if (!view.mapped(a) || buf.author != rank_) {
        view.bind_access_level(a, AL_PRIVATE);
        view.bind_author(a, buf.author);
        view.bind_committed(a, nullptr);
      }

      /* take ownership */
      view.bind_owner(a, rank_);
    } else
      LOGLN_OS("CTX pulled reserved=" << buf.p);

    /* prepare output */
    res = buf.p;

    return res;
  }

  template <typename T>
  inline void local_load(T *lp, uint64_t a) {
    LOGLN("CTX load %p size=%zu %llu", lp, sizeof(T), a);
    assert(view.committed(a) != nullptr);
    *lp = *reinterpret_cast<T *>(view.committed(a)->get());
  }

  inline unsigned long long local_rc_get(uint64_t a) { return mc.rc_get(a); }

  /*
   * actually transfer the private pointer to local memory
   *
   * @param p global pointer to be withdrawn
   *
   * @retval a raw pointer to the local memory for p
   */
  template <typename T>
  inline T *withdraw(const GlobalPointer &p) {
    assert(p.is_address());
    LOGLN_OS("CTX withdraw=" << p);
    uint64_t a = p.address();
    assert(view.access_level(a) == AL_PRIVATE);
    assert(!am_author(p));
    assert(am_owner(p));

    /* allocate backend memory */
    assert(view.committed(a) == nullptr);
    T *tmp = (T *)local_new<T>();
    using bp_t = backend_typed_ptr<T, void (*)(T *)>;
    bp_t *bp = local_new<bp_t>(tmp, DELETE<T>);

    /* bind parenthood */
    T *child = bp->typed_get();
    view.bind_parent(child, a);
    view.bind_child(a, child);

    /* issue remote load */
    forward_load(child, p);

    /* take ownership */
    view.bind_committed(a, bp);
    view.bind_author(a, rank_);

    return bp->typed_get();
  }

  template <typename T>
  void recv_kernel(T *lp, executor_id to, std::true_type) {
    local_links->raw_recv(lp, sizeof(T), to);
  }

  template <typename T>
  void recv_kernel(T *lp, executor_id to, std::false_type) {
    lp->ingest(
        [&](void *dst, size_t size) { local_links->raw_recv(dst, size, to); });
  }

  unsigned long long recv_rc(executor_id to) {
    unsigned long long res;
    local_links->raw_recv(&res, sizeof(unsigned long long), to);
    return res;
  }

  template <typename T>
  void forward_load(T *lp, const GlobalPointer &p) {
    assert(p.is_address());
    uint64_t a = p.address();
    executor_id to = view.author(a);
    LOGLN("CTX fwd LOAD size=%zu %llu dest=%lu", sizeof(T), a, to);

    /* send remote-load request */
    daemon_pointer dp;
    dp.op = daemon_pointer::RLOAD;
    dp.p = p;
    dp.size = sizeof(T);
    dp.from = rank_;
    local_links->send(dp, to);

    recv_kernel(lp, to, std::is_trivially_copyable<T>{});
  }

  unsigned long long forward_rc(const GlobalPointer &p) {
    assert(p.is_address());
    uint64_t a = p.address();
    executor_id to = view.author(a);
    LOGLN("CTX fwd RC %llu dest=%lu", a, to);

    /* send remote-rc request */
    daemon_pointer dp;
    dp.op = daemon_pointer::RC_GET;
    dp.p = p;
    dp.from = rank_;
    local_links->send(dp, to);

    return recv_rc(to);
  }

  inline void forward_inc(const GlobalPointer &p) {
    assert(p.is_address());
    uint64_t a = p.address();
    executor_id dest = view.author(a);
    LOGLN("CTX fwd +1 %llu dest=%lu", a, dest);
    daemon_pointer dp;
    dp.op = daemon_pointer::RC_INC;
    dp.from = rank_;
    dp.p = p;
    local_links->send(dp, dest);
  }

  inline void forward_dec(const GlobalPointer &p) {
    assert(p.is_address());
    uint64_t a = p.address();
    executor_id dest = view.author(a);
    LOGLN("CTX fwd -1 %llu dest=%lu", a, dest);
    daemon_pointer dp;
    dp.op = daemon_pointer::RC_DEC;
    dp.from = rank_;
    dp.p = p;
    local_links->send(dp, dest);
  }
};

#if __cplusplus >= 201703L
class Context_ {
 public:
  Context *ctx() { return &ctx_; }

 private:
  static inline Context ctx_
};
#else
class Context_ {
 public:
  static Context *ctx() {
    static Context ctx_;
    return &ctx_;
  }
};
#endif

static inline Context &ctx() { return *Context_::ctx(); }

/*
 * shortcuts
 */
template <typename obj_t, typename... Params>
inline obj_t *NEW(Params... p) {
  return ctx().local_new<obj_t>(p...);
}

template <typename T>
inline void DELETE(T *ptr) {
  return ctx().local_delete(ptr);
}

} /* namespace gam */

#endif /* CONTEXT_HPP_ */

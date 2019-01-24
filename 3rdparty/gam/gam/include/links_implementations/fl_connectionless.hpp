/* ***************************************************************************
 *
 *  This file is part of gam.
 *
 *  gam is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gam is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with gam. If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************
 */

/**
 * @file fl_connectionless.hpp
 * @author Maurizio Drocco
 * @brief implements fl_connectionless class
 *
 * @ingroup internals
 *
 * fl_connectionless implements gam communication primitives
 * on top of connection-less libfabric API.
 */

#ifndef GAM_INCLUDE_LINKS_IMPLEMENTATIONS_FL_CONNECTIONLESS_HPP_
#define GAM_INCLUDE_LINKS_IMPLEMENTATIONS_FL_CONNECTIONLESS_HPP_

#include <vector>
#include <algorithm>
#include <cassert>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include "../defs.hpp"
#include "GlobalPointer.hpp"
#include "Logger.hpp"

#include "fl_common.hpp"

namespace gam {

static struct fi_av_attr av_attr;
static struct fid_av *av; //AV table

class fl_connectionless {
public:
    fl_connectionless(executor_id cardinality, executor_id self, //
            const char *, size_t)
            : rank_to_addr(cardinality), self(self)
    {

    }

    static void init_links(char *src_node)
    {
        int ret = 0;

        fl_node(src_node);

        //query fabric contexts
#ifdef GAM_LOG
        fprintf(stderr, "LKS init_links\n");
#endif
        uint64_t flags = 0;
        fl_getinfo(&fl_info_, src_node, NULL, flags, FI_EP_RDM,
                FI_DIRECTED_RECV);

        fl_init(fl_info_);

        //init AV
        ret += fi_av_open(fl_domain_, &av_attr, &av, NULL);

        DBGASSERT(!ret);
    }

    static void fini_links()
    {
        int ret = 0;

        ret += fi_close(&av->fid);
        DBGASSERT(!ret);

        fl_fini();
        fi_freeinfo(fl_info_);
    }

    /*
     * add send link
     */
    void add(executor_id i, char *node, char *svc)
    {
        LOGLN("LKS @%p adding SEND to=%llu node=%s svc=%s", this, i, node, svc);

        // translate the address
        struct fi_info *fi;
        fl_dst_addr(node, svc, &fi, 0);

        //insert address to AV
        fi_addr_t fi_addr;
        int ret = fi_av_insert(av, fi->dest_addr, 1, &fi_addr, 0, NULL);
        assert(ret == 1);

        //map rank to av index
        rank_to_addr[i] = fi_addr;

        char addr[128], buf[128];
        unsigned long int len = 128;
        fi_av_lookup(av, fi_addr, (void *) addr, &len);
        assert(len < 128);
        len = 128;
        fi_av_straddr(av, addr, buf, &len);
        assert(len < 128);
        LOGLN("LKS @%p mapping rank=%llu -> addr=%llu (%s)", this, i, fi_addr,
                buf);
    }

    /*
     * add receive link
     */
    void add(char *node, char *svc)
    {
        LOGLN("LKS @%p adding RECV node=%s svc=%s", this, node, svc);
        init_endpoint(node, svc);
    }

    void finalize()
    {
        int ret = FI_SUCCESS;

        ret += fi_close(&ep_->fid);
        ret += fi_close(&rxcq->fid);
        ret += fi_close(&txcq->fid);

        DBGASSERT(!ret);
    }

    /*
     ***************************************************************************
     *
     * blocking send/receive
     *
     ***************************************************************************
     */
    void broadcast(const void *p, size_t size)
    {
        ssize_t ret = 0;
        executor_id to;
        for (to = 0; to < self; ++to)
            ret += fl_tx(ep_, txcq, p, size, rank_to_addr[to]);
        for (to = self + 1; to < rank_to_addr.size(); ++to)
            ret += fl_tx(ep_, txcq, p, size, rank_to_addr[to]);
        DBGASSERT(!ret);
    }

    void raw_send(const void *p, const size_t size, const executor_id to)
    {
        ssize_t ret = fl_tx(ep_, txcq, p, size, rank_to_addr[to]);
        DBGASSERT(!ret);
    }

    void raw_recv(void *p, const size_t size, const executor_id from)
    {
        ssize_t ret = rx(p, size, rank_to_addr[from]);
        DBGASSERT(!ret);
    }

    void raw_recv(void *p, const size_t size)
    {
        ssize_t ret = rx(p, size, FI_ADDR_UNSPEC);
        DBGASSERT(!ret);
    }

    /*
     ***************************************************************************
     *
     * non-blocking send/receive
     *
     ***************************************************************************
     */
    void nb_recv(void *p, const size_t size)
    {
        ssize_t ret = fl_post_rx(ep_, p, size, FI_ADDR_UNSPEC);
        DBGASSERT(!ret);
    }

    bool nb_poll()
    {
        struct fi_cq_entry cq_entry;
        ssize_t ret = fi_cq_read(rxcq, &cq_entry, 1); //non-blocking pop
        if (ret > 0)
            return true;
        else
        {
            DBGASSERT(ret == -FI_EAGAIN);
            return false;
        }
    }

private:
    struct fid_ep *ep_ = nullptr; //end point
    struct fid_cq *txcq = nullptr, *rxcq = nullptr; //completion queues

    std::vector<fi_addr_t> rank_to_addr;
    executor_id self;

    void init_endpoint(char *node, char *service)
    {
        int ret = FI_SUCCESS;

        //get fabric context
#ifdef GAM_LOG
        fprintf(stderr, "LKS src-endpoint node=%s svc=%s\n", node, service);
#endif
        fi_info *fi;
        fl_getinfo(&fi, node, service, FI_SOURCE, FI_EP_RDM, FI_DIRECTED_RECV);

        struct fi_cq_attr cq_attr;

        //create endpoint
        ret += fi_endpoint(fl_domain_, fi, &ep_, NULL);

        //prepare CQ attributes
        memset(&cq_attr, 0, sizeof(fi_cq_attr));
        cq_attr.format = FI_CQ_FORMAT_CONTEXT;
        cq_attr.wait_obj = FI_WAIT_NONE; //async

        //init TX CQ and bind
        cq_attr.size = fi->tx_attr->size;
        ret += fi_cq_open(fl_domain_, &cq_attr, &txcq, NULL);
        ret += fi_ep_bind(ep_, &txcq->fid, FI_SEND);

        //init RX CQ and bind
        cq_attr.size = fi->rx_attr->size;
        ret += fi_cq_open(fl_domain_, &cq_attr, &rxcq, NULL);
        ret += fi_ep_bind(ep_, &rxcq->fid, FI_RECV);

        //bind EP to AV
        ret += fi_ep_bind(ep_, &av->fid, 0);

        ret += fi_enable(ep_);
        DBGASSERT(!ret);

        //clean-up
        fi_freeinfo(fi);
    }

    ssize_t rx(void *rx_buf, size_t size, fi_addr_t from)
    {
        ssize_t ret = 0;

        //recv
        ret += fl_post_rx(ep_, rx_buf, size, from);

        //wait on RX CQ
        ret += fl_spin_for_comp(rxcq);

        return ret;
    }
}
;

} /* namespace gam */

#endif /* GAM_INCLUDE_LINKS_IMPLEMENTATIONS_FL_CONNECTIONLESS_HPP_ */

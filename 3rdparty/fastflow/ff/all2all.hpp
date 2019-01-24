/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file all2all.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow all-2-all building block
 *
 * @detail FastFlow basic contanier for a shared-memory parallel activity 
 *
 */

#ifndef FF_A2A_HPP
#define FF_A2A_HPP

/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */

#include <ff/node.hpp>
#include <ff/multinode.hpp>

namespace ff {
        
class ff_a2a: public ff_node {
    friend class ff_farm;
    friend class ff_pipeline;    
protected:
    inline int cardinality(BARRIER_T * const barrier)  { 
        int card=0;
        for(size_t i=0;i<workers1.size();++i) 
            card += workers1[i]->cardinality(barrier);
        for(size_t i=0;i<workers2.size();++i) 
            card += workers2[i]->cardinality(barrier);
        
        return card;
    }
    inline int cardinality() const { 
        int card=0;
        for(size_t i=0;i<workers1.size();++i) card += workers1[i]->cardinality();
        for(size_t i=0;i<workers2.size();++i) card += workers2[i]->cardinality();
        return card;
    }

    inline int prepare() {

        size_t nworkers1 = workers1.size();
        size_t nworkers2 = workers2.size();

        // NOTE: the nodes of the second set will have in input a multi-producer queue
        //       if reduce_channels=true. The node itself is multi-input because otherwise we have
        //       problems with the blocking stuff.
        if (reduce_channels) {
            int ondemand = workers1[0]->ondemand_buffer();  // NOTE: here we suppose that all nodes in workers1 are homogeneous!
            for(size_t i=0;i<nworkers2; ++i) {
                // NOTE: if reduce_channels is set, each worker2 node has to wait nworkers1 EOS messages before
                //       terminating
                ff_node* t = new ff_buffernode(ondemand?ondemand:in_buffer_entries,ondemand?true:fixedsize, i, nworkers1);
                assert(t);
                internalSupportNodes.push_back(t);
                workers2[i]->set_input(t);                
                for(size_t j=0;j<nworkers1;++j)                     
                    workers1[j]->set_output(t);
            }
        } else {
            int ondemand = workers1[0]->ondemand_buffer();  // NOTE: here we suppose that all nodes in workers1 are homogeneous!
            for(size_t i=0;i<nworkers2; ++i) {
                for(size_t j=0;j<nworkers1;++j) {
                    // NOTE: if reduce_channels is set, each worker2 node has to wait nworkers1 EOS messages before
                    //       terminating
                    ff_node* t = new ff_buffernode(ondemand?ondemand:in_buffer_entries,ondemand?true:fixedsize, j);
                    assert(t);
                    internalSupportNodes.push_back(t);                    
                    workers1[j]->set_output(t);
                    workers2[i]->set_input(t);
                }
            }
        }
        if (outputNodes.size()) {
            if (outputNodes.size() != workers2.size()) {
                error("A2A, prepare, invalid state\n");
                return -1;
            }
        }
        for(size_t i=0;i<outputNodes.size(); ++i) {
            assert(!workers2[i]->isMultiOutput());
            assert(outputNodes[i]->get_in_buffer() != nullptr);            
            assert(workers2[i]->get_out_buffer() == nullptr);
            if (workers2[i]->set_output_buffer(outputNodes[i]->get_in_buffer()) <0) 
                return -1;
        }


        // blocking stuff
        pthread_mutex_t   *m        = NULL;
        pthread_cond_t    *c        = NULL;
        std::atomic_ulong *counter  = NULL;
        for(size_t i=0;i<nworkers2;++i) {
            // initialize worker2 local cons_* stuff and sets all p_cons_*
            if (!workers2[i]->init_input_blocking(m,c,counter)) return -1;
        }
        for(size_t i=0;i<nworkers1;++i) {
            // initialize worker1 local prod_* stuff and sets all p_prod_*
            if (!workers1[i]->init_output_blocking(m,c,counter)) return -1;
        }
            
        prepared = true;
        return 0;
    }

    void *svc(void*) { return FF_EOS; }

    
public:
    enum {DEF_IN_BUFF_ENTRIES=2048};

    /**
     *
     * reduce_channels should be set to true only if we want/need to reduce the number of channel
     * by making use of multi-producer input queues.
     *
     */
    ff_a2a(bool reduce_channels=false, int in_buffer_entries=DEF_IN_BUFF_ENTRIES,
           bool fixedsize=false):prepared(false),fixedsize(fixedsize),
                                 reduce_channels(reduce_channels), 
                                 in_buffer_entries(in_buffer_entries)
    {}
    
    virtual ~ff_a2a() {
        if (barrier) delete barrier;
        if (workers1_to_free) {
            for(size_t i=0;i<workers1.size();++i) {
                delete workers1[i];
                workers1[i] = nullptr;
            }
        }
        if (workers2_to_free) {
            for(size_t i=0;i<workers2.size();++i) {
                delete workers2[i];
                workers2[i] = nullptr;
            }
        }
        for(size_t i=0;i<internalSupportNodes.size();++i) 
            delete internalSupportNodes[i];
    }

    /**
     * The nodes of the first set must be either standard ff_node or a node that is multi-output, 
     * e.g., a composition where the last stage is a multi-output node
     * 
     */
    int add_firstset(const std::vector<ff_node *> & w, int ondemand=0, bool cleanup=false) {
        if (w.size()==0) {
            error("A2A, try to add zero workers to the first set!\n");
            return -1; 
        }        
        if (w[0]->isFarm() || w[0]->isAll2All()) {
            error("A2A, try to add invalid nodes to the first set\n");
            return -1;
        }
        if (w[0]->isMultiOutput()) {  // supposing all are of the same type
            for(size_t i=0;i<w.size();++i) {
                if (ondemand && (w[i]->ondemand_buffer()==0))
                    w[i]->set_scheduling_ondemand(ondemand);
                workers1.push_back(w[i]);
                (workers1.back())->set_id(int(i));
            }
            workers1_to_free = cleanup;
            return 0;
        }
        // the nodes in the first set cannot be multi-input nodes without being
        // also multi-output
        if (w[0]->isMultiInput()) {
            error("A2A, the nodes of the first set cannot be multi-input nodes without being also multi-output (i.e., a composition of nodes). The node must be either standard node or multi-output node or compositions where the second stage is a multi-output node\n");
            return -1;
        }
        // it is a standard node, so we transform it to a multi-output node
        for(size_t i=0;i<w.size();++i) {
            ff_monode *mo = new mo_transformer(w[i], cleanup);
            if (!mo) {
                error("A2A, FATAL ERROR not enough memory\n");
                return -1;
            }
            mo->set_id(int(i));
            mo->set_scheduling_ondemand(ondemand);
            workers1.push_back(mo);
        }
        workers1_to_free = true;
        return 0;
    }
    
    /**
     * The nodes of the second set must be either standard ff_node or a node that is multi-input.
     * 
     */
    int add_secondset(const std::vector<ff_node *> & w, bool cleanup=false) {
        if (w.size()==0) {
            error("A2A, try to add zero workers to the second set!\n");
            return -1; 
        }        
        if (w[0]->isFarm() || w[0]->isAll2All()) {
            error("A2A, try to add invalid nodes to the second set \n");
            return -1;
        }
        if (w[0]->isMultiInput()) { // we suppose that all others are the same
            for(size_t i=0;i<w.size();++i) {
                workers2.push_back(w[i]);
                (workers2.back())->set_id(int(i));
            }
            workers2_to_free = cleanup;
        } else {
            if (w[0]->isMultiOutput()) {
                error("A2A, the nodes of the second set cannot be multi-output nodes without being also multi-input (i.e., a composition of nodes). The node must be either standard node or multi-input node or compositions where the first stage is a multi-input node\n");
                return -1;
            }

            // here we have to transform the standard node into a multi-input node
            for(size_t i=0;i<w.size();++i) {
                ff_minode *mi = new mi_transformer(w[i], cleanup);
                if (!mi) {
                    error("A2A, FATAL ERROR not enough memory\n");
                    return -1;
                }
                mi->set_id(int(i));
                workers2.push_back(mi);
            }
            workers2_to_free = true;
        }
        return 0;
    }

    bool reduceChannel() const { return reduce_channels;}
    
    void skipfirstpop(bool sk)   { 
        for(size_t i=0;i<workers1.size(); ++i)
            workers1[i]->skipfirstpop(sk);
        skip1pop=sk;
    }

    void blocking_mode(bool blk=true) {
        blocking_in = blocking_out = blk;
    }

    void no_mapping() {
        default_mapping = false;
    }
    
    void no_barrier() {
        initial_barrier=false;
    }
    
    int run(bool skip_init=false) {
        if (!skip_init) {        
#if defined(FF_INITIAL_BARRIER)
            if (initial_barrier) {
                // set the initial value for the barrier 
                if (!barrier)  barrier = new BARRIER_T;
                const int nthreads = cardinality(barrier);
                if (nthreads > MAX_NUM_THREADS) {
                    error("PIPE, too much threads, increase MAX_NUM_THREADS !\n");
                    return -1;
                }
                barrier->barrierSetup(nthreads);
            }
#endif
            skipfirstpop(true);
        }
        if (!prepared) if (prepare()<0) return -1;
        
        const size_t nworkers1 = workers1.size();
        const size_t nworkers2 = workers2.size();
        
        for(size_t i=0;i<nworkers1; ++i) {
            workers1[i]->blocking_mode(blocking_in);
            if (!default_mapping) workers1[i]->no_mapping();
            if (workers1[i]->run(true)<0) {
                error("ERROR: A2A, running worker (first set) %d\n", i);
                return -1;
            }
        }
        for(size_t i=0;i<nworkers2; ++i) {
            workers2[i]->blocking_mode(blocking_in);
            if (!default_mapping) workers1[i]->no_mapping();
            if (workers2[i]->run(true)<0) {
                error("ERROR: A2A, running worker (second set) %d\n", i);
                return -1;
            }
        }
        return 0;
    }
    
    int wait() {
        int ret=0;
        const size_t nworkers1 = workers1.size();
        const size_t nworkers2 = workers2.size();
        for(size_t i=0;i<nworkers1; ++i)
            if (workers1[i]->wait()<0) {
                error("A2A, waiting Worker1 thread, id = %d\n",workers1[i]->get_my_id());
                ret = -1;
            } 
        for(size_t i=0;i<nworkers2; ++i)
            if (workers2[i]->wait()<0) {
                error("A2A, waiting Worker2 thread, id = %d\n",workers2[i]->get_my_id());
                ret = -1;
            } 
        
        return ret;
    }
    
    int run_and_wait_end() {
        if (isfrozen()) {  // TODO 
            error("A2A: Error: feature not yet supported\n");
            return -1;
        } 
        if (run()<0) return -1;
        if (wait()<0) return -1;
        return 0;
    }

    const svector<ff_node*>& getFirstSet()  const { return workers1; }
    const svector<ff_node*>& getSecondSet() const { return workers2; }

    int numThreads() const { return cardinality(); }

    int set_output(const svector<ff_node *> & w) {
        if (outputNodes.size()+w.size() > workers2.size()) return -1;
        outputNodes +=w;
        return 0; 
    }

    int set_output(ff_node *node) {
        if (outputNodes.size()+1 > workers2.size()) return -1;
        outputNodes.push_back(node); 
        return 0;
    }

    
    void get_out_nodes(svector<ff_node*>&w) {
        for(size_t i=0;i<workers2.size();++i)
            workers2[i]->get_out_nodes(w);
        if (w.size() == 0)
            w += getSecondSet();
    }
    void get_in_nodes(svector<ff_node*>&w) {
        for(size_t i=0;i<workers1.size();++i)
            workers1[i]->get_in_nodes(w);
        if (w.size() == 0)
            w += getFirstSet();
    }


    /**
     * \brief Feedback channel (pattern modifier)
     * 
     * The last stage output stream will be connected to the first stage 
     * input stream in a cycle (feedback channel)
     */
    int wrap_around() {

        if (workers2[0]->isMultiOutput()) { // we suppose that all others are the same
            if (workers1[0]->isMultiInput()) { // we suppose that all others are the same
                for(size_t i=0;i<workers2.size(); ++i) {
                    for(size_t j=0;j<workers1.size();++j) {
                        ff_node* t = new ff_buffernode(in_buffer_entries,false);
                        t->set_id(i);
                        internalSupportNodes.push_back(t);
                        workers2[i]->set_output_feedback(t);
                        workers1[j]->set_input_feedback(t);
                    }                    
                }
            } else {
                // the cardinatlity of the first and second set of workers must be the same
                if (workers1.size() != workers2.size()) {
                    error("A2A, wrap_around, the workers of the second set are not multi-output nodes so the cardinatlity of the first and second set must be the same\n");
                    return -1;
                }

                if (create_input_buffer(in_buffer_entries, false) <0) {
                    error("A2A, error creating input buffers\n");
                    return -1;
                }
                
                for(size_t i=0;i<workers2.size(); ++i)
                    workers2[i]->set_output_feedback(workers1[i]);
                               
            }
            skipfirstpop(true);                
        } else {
            // the cardinatlity of the first and second set of workers must be the same
            if (workers1.size() != workers2.size()) {
                error("A2A, wrap_around, the workers of the second set are not multi-output nodes so the cardinatlity of the first and second set must be the same\n");
                return -1;
            }

            if (!workers1[0]->isMultiInput()) {  // we suppose that all others are the same
                if (create_input_buffer(in_buffer_entries, false) <0) {
                    error("A2A, error creating input buffers\n");
                    return -1;
                }
                
                for(size_t i=0;i<workers2.size(); ++i)
                    workers2[i]->set_output_buffer(workers1[i]->get_in_buffer());
                
            } else {
                
                if (create_output_buffer(in_buffer_entries, false) <0) {
                    error("A2A, error creating output buffers\n");
                    return -1;
                }
                
                for(size_t i=0;i<workers1.size(); ++i)
                    if (workers1[i]->set_input_feedback(workers2[i])<0) {
                        error("A2A, wrap_around, the nodes of the first set are not multi-input\n");
                        return -1;
                    }
            }
        }

        // blocking stuff....
        pthread_mutex_t   *m        = NULL;
        pthread_cond_t    *c        = NULL;
        std::atomic_ulong *counter  = NULL;
        if (workers2[0]->isMultiOutput()) {
            for(size_t i=0;i<workers2.size(); ++i) {
                if (!workers2[i]->init_output_blocking(m,c,counter)) return -1;
            }
        } else {
            assert(workers1.size() == workers2.size());
            for(size_t i=0;i<workers2.size(); ++i) {
                if (!workers2[i]->init_output_blocking(m,c,counter)) return -1;
                workers1[i]->set_input_blocking(m,c,counter);
            }
        }
        if (workers1[0]->isMultiInput()) {
            for(size_t i=0;i<workers1.size();++i) {
                if (!workers1[i]->init_input_blocking(m,c,counter)) return -1;
            }
        } else {
            assert(workers1.size() == workers2.size());
            for(size_t i=0;i<workers1.size();++i) {
                if (!workers1[i]->init_input_blocking(m,c,counter)) return -1;
                workers2[i]->set_output_blocking(m,c,counter);
            }
        }
        
        return 0;
    }

    // time functions --------------------------------

    const struct timeval  getstarttime()  const {
        const struct timeval zero={0,0};        
        std::vector<struct timeval > workertime(workers1.size(),zero);
        for(size_t i=0;i<workers1.size();++i)
            workertime[i]=workers1[i]->getstarttime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }
    const struct timeval  getwstartime()  const {
        const struct timeval zero={0,0};        
        std::vector<struct timeval > workertime(workers1.size(),zero);
        for(size_t i=0;i<workers1.size();++i)
            workertime[i]=workers1[i]->getwstartime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }
    
    
    const struct timeval  getstoptime()  const {
        const struct timeval zero={0,0};        
        std::vector<struct timeval > workertime(workers2.size(),zero);
        for(size_t i=0;i<workers2.size();++i)
            workertime[i]=workers2[i]->getstoptime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }
    
    const struct timeval  getwstoptime() const {
        const struct timeval zero={0,0};
        std::vector<struct timeval > workertime(workers2.size(),zero);
        for(size_t i=0;i<workers2.size();++i) {
            workertime[i]=workers2[i]->getwstoptime();
        }
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }
    
    double ffTime() { return diffmsec(getstoptime(),getstarttime()); }

    double ffwTime() { return diffmsec(getwstoptime(),getwstartime()); }

#if defined(TRACE_FASTFLOW)
    void ffStats(std::ostream & out) { 
        out << "--- a2a:\n";
        out << "--- L-Workers:\n";
        for(size_t i=0;i<workers1.size();++i) workers1[i]->ffStats(out);
        out << "--- R-Workers:\n";
        for(size_t i=0;i<workers2.size();++i) workers2[i]->ffStats(out);
    }
#else
    void ffStats(std::ostream & out) { 
        out << "FastFlow trace not enabled\n";
    }
#endif

    
    
protected:
    bool isMultiInput()  const { return true;}
    bool isMultiOutput() const { return true;}
    bool isAll2All()     const { return true; }    

    int create_input_buffer(int nentries, bool fixedsize) {
        size_t nworkers1 = workers1.size();
        for(size_t i=0;i<nworkers1; ++i)
            if (workers1[i]->create_input_buffer(nentries,fixedsize)==-1) return -1;
        return 0;
    }

    int create_input_buffer_mp(int nentries, bool fixedsize, int neos=1) {
        size_t nworkers1 = workers1.size();
        for(size_t i=0;i<nworkers1; ++i) {
            if (workers1[i]->create_input_buffer_mp(nentries,fixedsize, neos)==-1) return -1;
        }
        return 0;
    }

    int create_output_buffer(int nentries, bool fixedsize=false) {
        size_t nworkers2 = workers2.size();
        for(size_t i=0;i<nworkers2; ++i) {
            if (workers2[i]->isMultiOutput()) {
                ff_node* t = new ff_buffernode(nentries,fixedsize); 
                t->set_id(i);
                internalSupportNodes.push_back(t);
                workers2[i]->set_output(t);
            } else
                if (workers2[i]->create_output_buffer(nentries,fixedsize)==-1) return -1;
        }
        return 0;        
    }
    
    bool init_input_blocking(pthread_mutex_t   *&m,
                             pthread_cond_t    *&c,
                             std::atomic_ulong *&counter) {
        size_t nworkers1 = workers1.size();
        for(size_t i=0;i<nworkers1; ++i) {
            pthread_mutex_t   *m        = NULL;
            pthread_cond_t    *c        = NULL;
            std::atomic_ulong *counter  = NULL;
            if (!workers1[i]->init_input_blocking(m,c,counter)) return false;
        }
        return true;
    }
    bool init_output_blocking(pthread_mutex_t   *&m,
                              pthread_cond_t    *&c,
                              std::atomic_ulong *&counter) {
        size_t nworkers2 = workers2.size();
        for(size_t i=0;i<nworkers2; ++i) {
            pthread_mutex_t   *m        = NULL;
            pthread_cond_t    *c        = NULL;
            std::atomic_ulong *counter  = NULL;
            if (!workers2[i]->init_output_blocking(m,c,counter)) return false;
        }
        return true;
    }
    void set_input_blocking(pthread_mutex_t   *&m,
                            pthread_cond_t    *&c,
                            std::atomic_ulong *&counter) {
        size_t nworkers1 = workers1.size();
        for(size_t i=0;i<nworkers1; ++i) {
            workers1[i]->set_input_blocking(m,c,counter);
        }
    }    
    void set_output_blocking(pthread_mutex_t   *&m,
                             pthread_cond_t    *&c,
                             std::atomic_ulong *&counter) {
        size_t nworkers2 = workers2.size();
        for(size_t i=0;i<nworkers2; ++i) {
            workers2[i]->init_output_blocking(m,c,counter);
        }
    }
   
protected:
    bool workers1_to_free=false;
    bool workers2_to_free=false;
    bool prepared, fixedsize,reduce_channels;
    int in_buffer_entries;
    svector<ff_node*>  workers1;  // first set, nodes must be multi-output
    svector<ff_node*>  workers2;  // second set, nodes must be multi-input
    svector<ff_node*>  outputNodes;
    svector<ff_node*>  internalSupportNodes;
};

} // namespace ff 
#endif /* FF_A2A_HPP */

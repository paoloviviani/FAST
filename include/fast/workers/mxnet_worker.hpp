/*
 * worker.hpp
 *
 *  Created on: Aug 8, 2018
 *      Author: pvi
 */

#ifndef FAST_FAST_WORKERS_MXNET_WORKER_HPP_
#define FAST_FAST_WORKERS_MXNET_WORKER_HPP_

#include <array>
#include <vector>
#include <map>
#include <chrono>

#include "gff.hpp"
#include "gam.hpp"

#include <ff/pipeline.hpp>
#include <ff/node.hpp>

namespace FAST {

/**
 * Fastflow auxiliary stuff
 */
static auto TERMINATION_TAG = ff::FF_TAG_MIN;
static auto NEXT_ITERATION = (void *)((uint64_t)ff::FF_TAG_MIN + 1);
static auto END_OF_INPUT = (void *)((uint64_t)ff::FF_TAG_MIN + 3);

constexpr auto EOI_TOKEN = gff::go_on - 1;
constexpr auto TRIGGER_TOKEN = gff::go_on - 2;

template<typename T>
gam::public_ptr<T> token2public(uint64_t token) {
    return gam::public_ptr<T>(gam::GlobalPointer(token));
}

typedef std::vector<mxnet::cpp::NDArray> NDAvector;

/**
 * Nodes of the local pipeline
 * this pipeline executed in accelerator mode is used
 * to hide latencies related to:
 * - copy of data from public ptr to GPU/CPU memory
 * - training loop
 * - auxiliary node
 * - copy of data from GPU/CPU memory to public ptr
 */
template <typename ModelLogic, typename T >
class InputStage: public ff::ff_node {
public:

    InputStage(ModelLogic * logic) : logic_(logic), buffer_(nullptr), first_push_(true) {}

    void * svc(void * task) {

        if (task == NEXT_ITERATION) {
            FAST_DEBUG("(INPUT STAGE): got trigger");
            if (first_push_){
                this->ff_send_out( NEXT_ITERATION );
                first_push_ = false;
            }
            return ff::FF_GO_ON;
        }

        auto recv_ptr = (FAST::gam_vector<T> *)task;

        FAST_DEBUG("(INPUT STAGE): got real pointer of size " << (*recv_ptr).size())
        FAST::accumToNDVec( *recv_ptr, *buffer_, logic_->arg_names, logic_->data_tag, logic_->label_tag, 1., mxnet::cpp::Context::cpu() );
        recv_ptr->clear();
        gam::DELETE(recv_ptr);

        if (this->get_out_buffer()->empty()) {
            FAST_DEBUG("(INPUT STAGE): push gradients");
            this->ff_send_out((void *)buffer_);
            buffer_ = gam::NEW<NDAvector>();
            FAST::buildNDVec( *buffer_, logic_->exec->grad_arrays, logic_->arg_names, mxnet::cpp::Context::cpu() );
        }
        return ff::FF_GO_ON;
    }

    int svc_init() {
        FAST_DEBUG("(INPUT STAGE): init stage");
        buffer_ = gam::NEW<NDAvector>();
        FAST::buildNDVec( *buffer_, logic_->exec->grad_arrays, logic_->arg_names, mxnet::cpp::Context::cpu() );
        FAST_DEBUG("(INPUT STAGE): Built NDVec");
        return 0;
    }

    void svc_end() {
        if(buffer_)
            delete buffer_;
    }
private:
    ModelLogic * logic_;
    NDAvector * buffer_;
    bool first_push_ = true;
};

template <typename ModelLogic, typename T >
class TrainerStage: public ff::ff_minode {
public:

    TrainerStage(ModelLogic * logic) : logic_(logic) {}

    void * svc(void * task) {

        if (!logic_->max_epoch_reached) {
            if (task != NEXT_ITERATION) {
                // got a pointer from the input stage
                NDAvector * in_ptr = (NDAvector  *)task;
                logic_->update( *in_ptr );
                FAST_DEBUG("(TRAINER STAGE): executed batch from gradients");
                in_ptr->clear();
                gam::DELETE(in_ptr);
            }
            logic_->run_batch();
            FAST_DEBUG("(TRAINER STAGE): executed local batch ");
            return NEXT_ITERATION;
        }
        else {
            FAST_DEBUG("(TRAINER STAGE): returned end of input");
            return END_OF_INPUT;
        }
        return ff::FF_GO_ON;
    }
private:
    ModelLogic * logic_;

    void eosnotify(ssize_t id) {
        if (id == 0) {
            // got EOS from input - forward TERMINATION_TAG message
            this->ff_send_out(TERMINATION_TAG);
        } else {
            // got EOS from feedback - forward downstream to trigger termination
            this->ff_send_out(ff::FF_EOS);
            // got both EOSs - node will be terminated here
        }
    }
};

class internal_out_stage : public ff::ff_monode {
    void *svc(void *in) {
        if (in == TERMINATION_TAG){
            // send EOS to the feedback channel
            ff_send_out_to(ff::FF_EOS, 0);
        }
        else {
            // send a NEXT_ITERATION message to the feedback channel
            if (outnodes_[0]->get_out_buffer()->empty() && in != END_OF_INPUT)
                ff_send_out_to(NEXT_ITERATION, 0);
            // forward the input pointer downstream
            ff_send_out_to(in, 1);
        }
        return ff::FF_GO_ON;
    }

    int svc_init() {
        this->get_out_nodes(outnodes_);
        return 0;
    }
private:
    ff::svector<ff_node*> outnodes_;
};

template <typename ModelLogic, typename T >
class OutputStage: public ff::ff_node {
public:

    OutputStage(ModelLogic * logic) : logic_(logic){}

    void * svc(void * task) {
        if (task == END_OF_INPUT)
            return END_OF_INPUT;
        gam_vector<T> * out = gam::NEW<gam_vector<T>>();
        NDVecToVec( logic_->exec->grad_arrays, logic_->arg_names, *out, logic_->data_tag, logic_->label_tag, 0.25);
        FAST_DEBUG("(OUTPUT STAGE): serialized size " << out->size());
        return (void*)out;
    }

private:
    ModelLogic * logic_;
};

/**
 * Actual worker object, to be specialised based on the specific
 * business logic (included in ModelLogic) and on the specific
 * data type (float, int, bool)
 */
template< typename ModelLogic, typename T>
class MXNetWorkerLogic {

using public_raw_pair = std::pair< FAST::gam_vector<T> * , gam::public_ptr< FAST::gam_vector<T> > >;

public:

    gff::token_t svc(gam::public_ptr< gam_vector<T> > &in, gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
        // Check message and offload to pipe
        switch(in.get().address()) {
        case TRIGGER_TOKEN: {
            pipe_->offload(NEXT_ITERATION);
            break;
        } // scope of 'x' ends here
        case EOI_TOKEN: {
            FAST_INFO("Received EOI token");
            assert(eoi_cnt_ < ( neighbors ));
            if (!eoi_out)
                c.emit(token2public<FAST::gam_vector<T>>(EOI_TOKEN));
            if(++eoi_cnt_ == neighbors)
                return gff::eos;
            return gff::go_on;
        }
        default: { //data
            auto in_ptr = in.unique_local().release();
            in.reset();
            pipe_->offload( (void*)in_ptr );
        }
        }

        void * outptr = nullptr;
        while (!eoi_out) {
            pipe_->load_result(&outptr);
            if (outptr == END_OF_INPUT) {
                FAST_DEBUG("(MXNET WORKER): Got EOI");
                if (!eoi_out)
                    c.emit(token2public<FAST::gam_vector<T>>(EOI_TOKEN));
                eoi_out = true;
                return gff::go_on;
            }
            else { //out data
                FAST_DEBUG("(MXNET WORKER): Got data");
                FAST::gam_vector<T> * out_vec = (FAST::gam_vector<T> *)outptr;
                gam::public_ptr<FAST::gam_vector<T>> out_ptr(out_vec, gam::DELETE<FAST::gam_vector<T> >);

                c.emit(std::move(out_ptr));
                return gff::go_on;
            }
        }
        return gff::go_on;
    }

    void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {

        neighbors = c.internals.out_cardinality();

        pipe_ = new ff::ff_pipeline(true);
        training_ = new ff::ff_pipeline();

        logic_.init();
        FAST_DEBUG("(MXNET WORKER): Initialized model logic");

        FAST_DEBUG("(MXNET WORKER): Creating pipeline")
        pipe_->add_stage( new InputStage<ModelLogic, T>(&logic_) );
        training_->add_stage( new TrainerStage<ModelLogic, T>(&logic_) );
        training_->add_stage( new internal_out_stage() );
        training_->wrap_around();
        pipe_->add_stage(training_);
        pipe_->add_stage( new OutputStage<ModelLogic, T>(&logic_) );

        pipe_->cleanup_nodes();
        training_->cleanup_nodes();

        FAST_DEBUG("(MXNET WORKER): Launching pipeline");
        pipe_->run();

        FAST_DEBUG("(MXNET WORKER): Emitting trigger");
        c.emit(token2public<FAST::gam_vector<T>>(TRIGGER_TOKEN));
    }

    void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c) {
        FAST_DEBUG("(FINALIZATION)");
        pipe_->offload( ff::FF_EOS );
        if (pipe_->wait()<0) {
            FAST_DEBUG("(FINALIZATION): error waiting pipe");
        }
        logic_.finalize();
    }
private:
    ff::ff_pipeline * pipe_;
    ff::ff_pipeline * training_;
    ModelLogic logic_;
    int eoi_cnt_ = 0;
    bool eoi_out = false;
    int neighbors;
};

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */

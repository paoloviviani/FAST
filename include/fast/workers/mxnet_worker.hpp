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

namespace FAST
{

/**
 * Fastflow auxiliary stuff
 */
static auto TERMINATION_TAG = ff::FF_TAG_MIN;
static auto NEXT_ITERATION = (void *)((uint64_t)ff::FF_TAG_MIN + 1);
static auto END_OF_INPUT = (void *)((uint64_t)ff::FF_TAG_MIN + 3);

constexpr auto EOI_TOKEN = gff::go_on - 1;
constexpr auto TRIGGER_TOKEN = gff::go_on - 2;

template <typename T>
gam::public_ptr<T> token2public(uint64_t token)
{
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
template <typename ModelLogic, typename T>
class InputStage : public ff::ff_node
{
  public:
    InputStage(ModelLogic *logic) : logic_(logic), buffer_(nullptr), first_push_(true) {}

    void *svc(void *task)
    {

        if (task == NEXT_ITERATION)
        {
            FAST_DEBUG("(INPUT STAGE): got trigger");
            if (true)
            {
                this->ff_send_out(NEXT_ITERATION);
                first_push_ = false;
            }
            return ff::FF_GO_ON;
        }

        auto recv_ptr = (FAST::gam_vector<T> *)task;

        FAST_DEBUG("(INPUT STAGE): got real pointer of size " << (*recv_ptr).size())
        FAST::accumToNDVec(*recv_ptr, *buffer_, logic_->arg_names, logic_->data_tag, logic_->label_tag, 1., mxnet::cpp::Context::cpu());
        recv_ptr->clear();
        gam::DELETE((std::vector<T> *)recv_ptr);

        if (this->get_out_buffer()->empty())
        {
            FAST_DEBUG("(INPUT STAGE): push gradients");
            this->ff_send_out((void *)buffer_);
            buffer_ = new NDAvector();
            FAST::buildNDVec(*buffer_, logic_->exec->grad_arrays, logic_->arg_names, mxnet::cpp::Context::cpu());
        }
        return NEXT_ITERATION;
    }

    int svc_init()
    {
        FAST_DEBUG("(INPUT STAGE): init stage");
        buffer_ = new NDAvector();
        FAST::buildNDVec(*buffer_, logic_->exec->grad_arrays, logic_->arg_names, mxnet::cpp::Context::cpu());
        FAST_DEBUG("(INPUT STAGE): Built NDVec");
        return 0;
    }

    void svc_end()
    {
        if (buffer_)
        {
            buffer_->clear();
            delete buffer_;
        }
    }

  private:
    ModelLogic *logic_;
    NDAvector *buffer_;
    bool first_push_ = true;
};

template <typename ModelLogic, typename T>
class TrainerStage : public ff::ff_node
{
  public:
    TrainerStage(ModelLogic *logic) : logic_(logic) {}

    void *svc(void *task)
    {

        while (!logic_->max_epoch_reached)
        {
            logic_->run_batch();
            this->ff_send_out(NEXT_ITERATION);
            FAST_DEBUG("(TRAINER STAGE): executed local batch ");

            // get_in_buffer()->pop(&task);
            this->Pop(&task, -1);
            if (task == ff::FF_EOS)
                return ff::FF_EOS;
            if (task != NEXT_ITERATION)
            {
                // got a pointer from the input stage
                NDAvector *in_ptr = (NDAvector *)task;
                logic_->update(*in_ptr);
                FAST_INFO("(TRAINER STAGE): executed batch from gradients");
                in_ptr->clear();
                delete in_ptr;
                // gam::DELETE(in_ptr);
            }
        }

        FAST_DEBUG("(TRAINER STAGE): returned end of input");
        return END_OF_INPUT;
    }

  private:
    ModelLogic *logic_;
};

template <typename ModelLogic, typename T>
class OutputStage : public ff::ff_node
{
  public:
    OutputStage(ModelLogic *logic) : logic_(logic) {}

    void *svc(void *task)
    {
        if (task == END_OF_INPUT)
            return END_OF_INPUT;
        gam_vector<T> *out = gam::NEW<gam_vector<T>>();
        NDVecToVec(logic_->exec->grad_arrays, logic_->arg_names, *out, logic_->data_tag, logic_->label_tag, 0.25);
        FAST_DEBUG("(OUTPUT STAGE): serialized size " << out->size());
        return (void *)out;
    }

  private:
    ModelLogic *logic_;
};

/**
 * Actual worker object, to be specialised based on the specific
 * business logic (included in ModelLogic) and on the specific
 * data type (float, int, bool)
 */
template <typename ModelLogic, typename T>
class MXNetWorkerLogic
{
  public:
    gff::token_t svc(gam::public_ptr<gam_vector<T>> &in, gff::OutBundleBroadcast<gff::NondeterminateMerge> &c)
    {
        // Check message and offload to pipe
        switch (in.get().address())
        {
        case TRIGGER_TOKEN:
        {
            pipe_->offload(NEXT_ITERATION);
            break;
        }
        case EOI_TOKEN:
        {
            FAST_INFO("Received EOI token");
            eoi = true;
            if (FAST::rank() == 0)
            {
                if (++eoi_cnt_ == FAST::cardinality() - 1)
                {
                    for (int i = 1; i < FAST::cardinality(); i++)
                        token2public<FAST::gam_vector<T>>(EOI_TOKEN).push(i);
                    return gff::eot;
                }
            }
            else
                return gff::eot;

            return gff::go_on;
        }
        default:
        { //data
            assert(in.get().is_address());
            if (!eoi)
            {
                auto in_ptr = in.unique_local().release();
                pipe_->offload((void *)in_ptr);
            }
        }
        }

        void *outptr = nullptr;
        while (!eoi)
        {
            pipe_->load_result(&outptr);
            if (outptr == END_OF_INPUT)
            {
                FAST_DEBUG("(MXNET WORKER): Got EOI");
                if (!eoi)
                {
                    FAST_INFO("(MXNET WORKER) Sent EOI token");
                    if (FAST::rank() != 0)
                        token2public<FAST::gam_vector<T>>(EOI_TOKEN).push(0);
                    eoi = true;
                }
                return gff::go_on;
            }
            else
            { //out data
                FAST_DEBUG("(MXNET WORKER): Got data");
                FAST::gam_vector<T> *out_vec = (FAST::gam_vector<T> *)outptr;
                gam::public_ptr<FAST::gam_vector<T>> out_ptr(out_vec, gam::DELETE<FAST::gam_vector<T>>);
                c.emit(std::move(out_ptr));
                return gff::go_on;
            }
        }
        return gff::go_on;
    }

    void svc_init(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c)
    {

        neighbors = c.internals.out_cardinality();

        pipe_ = new ff::ff_pipeline(true);
        training_ = new ff::ff_pipeline();

        logic_.init();
        FAST_DEBUG("(MXNET WORKER): Initialized model logic");

        FAST_DEBUG("(MXNET WORKER): Creating pipeline")
        pipe_->add_stage(new InputStage<ModelLogic, T>(&logic_));
        pipe_->add_stage(new TrainerStage<ModelLogic, T>(&logic_));
        pipe_->add_stage(new OutputStage<ModelLogic, T>(&logic_));

        pipe_->cleanup_nodes();

        FAST_DEBUG("(MXNET WORKER): Launching pipeline");
        pipe_->run();

        FAST_DEBUG("(MXNET WORKER): Emitting trigger");
        if (FAST::rank() != 0)
            token2public<FAST::gam_vector<T>>(TRIGGER_TOKEN).push(0);
        else
        {
            int count = 0;
            while(count < FAST::cardinality() -1)
            {
                auto p = gam::pull_public<float>();
                count++;
            }
            for (int i = 0; i < FAST::cardinality(); i++)
                token2public<FAST::gam_vector<T>>(TRIGGER_TOKEN).push(i);
        }
        
    }

    void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c)
    {
        pipe_->offload(ff::FF_EOS);
        FAST_DEBUG("(FINALIZATION)");
        float test_acc = logic_.val_acc.Get();
        auto accuracy = gam::make_public<float>(test_acc);
        if (FAST::rank() != 0)
            accuracy.push(0);
            
        FAST_INFO("(BEST WORKER): sent accuracy = " << test_acc);
        int best = FAST::rank();
        float max = test_acc;
        float acc = 0;
        if (FAST::rank() == 0)
        {
            for (int i = 1; i < FAST::cardinality(); i++)
            {
                auto p = gam::pull_public<float>(i);
                acc = *(p.local());
                if (max < acc)
                {
                    best = i;
                    max = acc;
                }
            }
            auto best_ptr = gam::make_public<int>(best);
            for (int i = 1; i < FAST::cardinality(); i++)
                best_ptr.push(i);
        }
        else
        {
            auto best_ptr = gam::pull_public<int>(0);
            best = *(best_ptr.local());
        }
        bool save = false;
        if (best == FAST::rank())
        {
            save = true;
            FAST_INFO("(BEST WORKER): " << best << "  accuracy = " << max);
        }
        logic_.finalize(save);
        if (pipe_->wait() < 0)
        {
            FAST_ERROR("(FINALIZATION): error waiting pipeline");
        }
    }

  private:
    ff::ff_pipeline *pipe_;
    ff::ff_pipeline *training_;
    ModelLogic logic_;
    int eoi_cnt_ = 0;
    bool eoi = false;
    int neighbors;
};

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */

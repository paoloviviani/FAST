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
    InputStage(ModelLogic *logic) : logic_(logic), buffer_(nullptr) {}

    void *svc(void *task)
    {

        if (task == NEXT_ITERATION)
        {
            FAST_DEBUG("(INPUT STAGE): got trigger");
            this->ff_send_out(NEXT_ITERATION);
            return ff::FF_GO_ON;
        }

        auto recv_ptr = (gam_vector<T> *)task;

        FAST_DEBUG("(INPUT STAGE): got real pointer of size " << (*recv_ptr).size())
        accumToNDVec(*recv_ptr, *buffer_, logic_->arg_names, logic_->data_tag, logic_->label_tag, mxnet::cpp::Context::cpu());
        recv_ptr->clear();
        gam::DELETE((std::vector<T> *)recv_ptr);

        FAST_DEBUG("(INPUT STAGE): push gradients");
        this->ff_send_out((void *)buffer_);
        buffer_ = new NDAvector();
        buildNDVec(*buffer_, logic_->exec->grad_arrays, logic_->arg_names, mxnet::cpp::Context::cpu());
        FAST_DEBUG("PUSHED");
        return NEXT_ITERATION;
    }

    int svc_init()
    {
        FAST_DEBUG("(INPUT STAGE): init stage");
        buffer_ = new NDAvector();
        buildNDVec(*buffer_, logic_->exec->grad_arrays, logic_->arg_names, mxnet::cpp::Context::cpu());
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
            // bool pop = this->Pop(&task);

            while (this->Pop(&task, 10) && task != NEXT_ITERATION)
            {
                if (task == ff::FF_EOS)
                    return ff::FF_EOS;
                // got a pointer from the input stage
                NDAvector *in_ptr = (NDAvector *)task;
                assert(in_ptr->size() > 0);
                logic_->update(*in_ptr);
                FAST_DEBUG("(TRAINER STAGE): executed batch from gradients");
                FAST_INFO("UPDATED: " << ++upd_count);
                in_ptr->clear();
                delete in_ptr;
            }
        }

        FAST_DEBUG("(TRAINER STAGE): returned end of input");
        return END_OF_INPUT;
    }

  private:
    ModelLogic *logic_;
    int upd_count = 0;
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
        NDVecToVec(logic_->exec->grad_arrays, logic_->arg_names, *out, logic_->data_tag, logic_->label_tag);
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
            if (!first_)
                pipe_->offload(NEXT_ITERATION);
            first_ = false;
            break;
        }
        case EOI_TOKEN:
        {
            FAST_INFO("Received EOI token");
            eoi = true;
            if (rank() == 0)
            {
                if (++eoi_cnt_ == cardinality() - 1)
                {
                    for (int i = 1; i < cardinality(); i++)
                        token2public<gam_vector<T>>(EOI_TOKEN).push(i);
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
                buffer.push_back(in);
                if (buffer.size() < neighbors)
                    return gff::go_on;
                else
                {
                    FAST_DEBUG("RECEIVED: " << ++recv_count);
                    for (auto item : buffer)
                    {
                        auto in_ptr = item.unique_local().release();
                        pipe_->offload((void *)in_ptr);
                    }
                    buffer.clear();
                }
            }
        }
        }

        void *outptr = nullptr;
        while (!eoi)
        {
            // Change -1 to a positive integer to provide asynchronicity
            if(!pipe_->load_result(&outptr, -1))
            {
                c.emit(token2public<gam_vector<T>>(TRIGGER_TOKEN));
                return gff::go_on;
            }

            if (outptr == END_OF_INPUT)
            {
                FAST_DEBUG("(MXNET WORKER): Got EOI");
                if (!eoi)
                {
                    FAST_INFO("(MXNET WORKER) Sent EOI token");
                    if (rank() != 0)
                        token2public<gam_vector<T>>(EOI_TOKEN).push(0);
                    eoi = true;
                }
                return gff::go_on;
            }
            else
            { //out data
                FAST_DEBUG("(MXNET WORKER): Got data");
                gam_vector<T> *out_vec = (gam_vector<T> *)outptr;
                gam::public_ptr<gam_vector<T>> out_ptr(out_vec, gam::DELETE<gam_vector<T>>);
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
        c.emit(token2public<gam_vector<T>>(TRIGGER_TOKEN));
    }

    void svc_end(gff::OutBundleBroadcast<gff::NondeterminateMerge> &c)
    {
        pipe_->offload(ff::FF_EOS);
        if (pipe_->wait() < 0)
        {
            FAST_ERROR("(FINALIZATION): error waiting pipeline");
        }
        FAST_DEBUG("(FINALIZATION)");
        float test_acc = logic_.val_acc;
        auto accuracy = gam::make_public<float>(test_acc);
        if (rank() == 0)
            for (int i = 1; i < cardinality(); i++)
                token2public<gam_vector<T>>(TRIGGER_TOKEN).push(i);
        else
        {
            auto p = gam::pull_public<float>(0);
            assert(p.get().address() == TRIGGER_TOKEN);
            accuracy.push(0);
        }
        
        FAST_INFO("(BEST WORKER): sent accuracy = " << test_acc);
        int best = rank();
        float max = test_acc;
        float acc = 0;
        if (rank() == 0)
        {
            for (int i = 1; i < cardinality(); i++)
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
            for (int i = 1; i < cardinality(); i++)
                best_ptr.push(i);
        }
        else
        {
            auto best_ptr = gam::pull_public<int>(0);
            best = *(best_ptr.local());
        }
        bool save = false;
        if (best == rank())
        {
            save = true;
            FAST_INFO("(BEST WORKER): " << best << "  accuracy = " << max);
        }
       
        logic_.finalize(save);

        if (rank() == best)
            for (int i = 0; i < cardinality(); i++)
            {
                if (i == best) continue;
                token2public<gam_vector<T>>(TRIGGER_TOKEN).push(i);
            }
        else
        {
            auto p = gam::pull_public<float>(best);
            assert(p.get().address() == TRIGGER_TOKEN);
        }

    }

  private:
    ff::ff_pipeline *pipe_;
    ff::ff_pipeline *training_;
    ModelLogic logic_;
    int eoi_cnt_ = 0;
    bool eoi = false;
    int neighbors;
    int recv_count = 0;
    bool first_ = false;
    std::vector<gam::public_ptr<gam_vector<T>>> buffer;
};

} // namespace FAST

#endif /* FAST_FAST_WORKERS_MXNET_WORKER_HPP_ */

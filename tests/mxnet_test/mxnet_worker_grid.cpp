/*
 * gff_unit_tests.cpp
 *
 *  Created on: Aug 3, 2018
 *      Author: pvi
 */

#include <iostream>
#include <catch.hpp>
#include <string>
#include <cassert>
#include <cmath>
#include <stdlib.h>

#include <gff.hpp>
#include <fast.hpp>
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

#define CATCH_CONFIG_MAIN

#define BATCH_SIZE 10

Context ctx = Context::cpu(); // Use CPU for training

struct Dummy
{
  std::vector<NDArray> grad_arrays;
};

struct DummyAcc
{
  float acc;
  float Get() {return acc;};
};

class ModelLogic
{
public:
  void init()
  {
    Context ctx = Context::cpu(); // Use CPU for training

    exec = new Dummy();
    arg_names.push_back("first");
    arg_names.push_back("second");
    arg_names.push_back("third");
    arg_names.push_back("fourth");

    for (size_t i = 0; i < arg_names.size(); ++i)
    {
      exec->grad_arrays.push_back(NDArray(Shape(BATCH_SIZE, 10), ctx));
      exec->grad_arrays[i] = 0.;
    }
    val_acc.acc = FAST::rank();
    FAST_INFO("Logic initialized");
  }

  void run_batch()
  {
    FAST_INFO("(LOGIC): run batch, iteration = " << iter_);
    for (size_t i = 0; i < arg_names.size(); ++i)
    {
      exec->grad_arrays[i] += 0.1;
    }
    NDArray::WaitAll();
    std::this_thread::sleep_for(std::chrono::milliseconds(100+ (FAST::rank()*10) ));
    iter_++;
    if (iter_ == 5)
      max_epoch_reached = true; // Terminate
  }

  void update(std::vector<mxnet::cpp::NDArray> &in)
  {
    REQUIRE(in.size() > 0);
    for (size_t i = 0; i < arg_names.size(); ++i)
    {
      exec->grad_arrays[i] += in[i];
    }
    NDArray::WaitAll();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    FAST_INFO("(LOGIC UPDATE): updated");
  }

  void finalize(bool save=false)
  {
  }

  Dummy *exec;
  vector<string> arg_names;
  unsigned int iter_ = 0;
  bool max_epoch_reached = false;
  const std::string data_tag = "X";
  const std::string label_tag = "label";
  DummyAcc val_acc;
};

typedef gff::Filter<gff::NondeterminateMerge, gff::OutBundleBroadcast<gff::NondeterminateMerge>,        //
                    gam::public_ptr<FAST::gam_vector<float>>, gam::public_ptr<FAST::gam_vector<float>>, //
                    FAST::MXNetWorkerLogic<ModelLogic, float>>
    MxNetWorker;

/*
 *******************************************************************************
 *
 * mains
 *
 *******************************************************************************
 */

TEST_CASE("MxNet worker grid test", "gam,gff,multi,mxnet")
{
  FAST_LOG_INIT
  FAST_INFO("TEST name: " << Catch::getResultCapture().getCurrentTestName());

 	size_t grid_h = 3;
	size_t grid_w = 3;
	size_t workers = grid_h * grid_w;

	// Row major ordering
	std::vector<std::vector<gff::NondeterminateMerge>> incoming_channels(grid_h);
	std::vector<std::vector<gff::OutBundleBroadcast<gff::NondeterminateMerge>>> outgoing_channels(grid_h);

	for (int i = 0; i < grid_h; i++)
	{
		for (int j = 0; j < grid_w; j++)
		{
			incoming_channels.at(i).emplace_back();
		}
	}

	for (unsigned int i = 0; i < grid_h; i++)
	{
		for (unsigned int j = 0; j < grid_w; j++)
		{
			outgoing_channels.at(i).emplace_back();
			// Add neighboring channels (i+1,j),(i-1,j),(i,j+1),(i,j-1) in torus topology

			unsigned int up, right, down, left;
			i == grid_h - 1 ? down = 0 : down = i + 1;
			i == 0 ? up = grid_h - 1 : up = i - 1;
			j == grid_w - 1 ? right = 0 : right = j + 1;
			j == 0 ? left = grid_w - 1 : left = j - 1;
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(up).at(j));
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(down).at(j));
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(i).at(right));
			outgoing_channels.at(i).at(j).add_comm(incoming_channels.at(i).at(left));
		}
	}

	for (unsigned int i = 0; i < grid_h; i++)
	{
		for (unsigned int j = 0; j < grid_w; j++)
		{
			gff::add(MxNetWorker(incoming_channels.at(i).at(j), outgoing_channels.at(i).at(j)));
		}
	}

	/* execute the network */
	gff::run();
}

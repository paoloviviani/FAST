/*
 * example_unit_test.cpp
 *
 *  Created on: May 30, 2018
 *      Author: viviani
 */

#include <iostream>
#include <catch.hpp>
#include "fast.hpp"
#include "mxnet-cpp/MxNetCpp.h"
#include <string>

using namespace std;
using namespace mxnet::cpp;
//using namespace FAST;

Context ctx = Context::cpu();  // Use CPU for training

TEST_CASE( "Gam basic", "gam" ) {
	FAST_LOG_INIT
	if (gam::rank() == 1) {
	auto p = gam::make_private<int>(42);
	assert(p != nullptr);
	auto local_p = p.local();
	REQUIRE(*local_p == 42);
	}
}
//
//TEST_CASE( "SPMD public vector ping-pong", "gam,vector,public" ) {
//	if (gam::cardinality() > 1) {
//		vector<int> ref = {1,2,3};
//		switch (gam::rank()) {
//		case 0:
//		{
//			// check local consistency on public pointer
//			std::shared_ptr<FAST::gam_vector<int>> lp = nullptr;
//			{
//				auto p = gam::make_public<FAST::gam_vector<int>>(ref);
//				lp = p.local();
//				// here end-of-scope triggers the destructor on the original object
//			}
//			REQUIRE(*lp == ref);
//
//			auto p = gam::make_public<FAST::gam_vector<int>>(ref);
//			REQUIRE(p.use_count() == 1);
//			p.push(1);
//			REQUIRE(p.use_count() == 2);
//			auto q = gam::make_private<FAST::gam_vector<int>>(ref);
//			q.push(1);
//			break;
//		}
//		case 1:
//		{
//			std::shared_ptr<FAST::gam_vector<int>> lp = nullptr;
//			{
//				auto p = gam::pull_public<FAST::gam_vector<int>>(0);
//				lp = p.local();
//				// here end-of-scope triggers the destructor on the original object
//			}
//			REQUIRE(*lp == ref);
//
//			auto q = gam::pull_private<FAST::gam_vector<int>>(0);
//			auto lq = q.local();
//			REQUIRE(*lq == ref);
//			break;
//		}
//		}
//	}
//}

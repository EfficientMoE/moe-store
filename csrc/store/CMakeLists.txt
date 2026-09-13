# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

cmake_minimum_required(VERSION 3.18)
project(moe_store_csrc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(moe_store_index STATIC index_v2.cc)
target_include_directories(moe_store_index PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(moe_store_index PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

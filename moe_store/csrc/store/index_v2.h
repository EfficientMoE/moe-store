// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

// C++ reader for the v2 store index (docs/store-format-v2.md §4).
// Torch-free and byte-compatible with the Python reference implementation
// in moe_store/index.py; cross-language compatibility is enforced by
// tests/test_cpp_index_compat.py.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moe_store {

inline constexpr char kIndexMagic[8] = {'M', 'O', 'E', 'S',
                                        'T', 'O', 'R', '2'};
inline constexpr std::uint32_t kIndexVersion = 2;
inline constexpr std::int64_t kGroupAlignment = 4096;
inline constexpr std::int64_t kMemberAlignment = 64;
inline constexpr const char* kIndexFileName = "store_index";
inline constexpr const char* kDataFilePrefix = "store_data_";

enum class GroupKind : std::uint8_t { kDense = 0, kExpert = 1 };

struct MemberMeta {
  std::uint32_t tensor_id = 0;
  std::string name;
  std::int64_t rel_offset = 0;
  std::int64_t size = 0;
  std::string dtype;
  std::vector<std::int64_t> shape;
};

struct GroupMeta {
  std::uint64_t group_id = 0;
  GroupKind kind = GroupKind::kDense;
  std::int32_t layer_id = -1;
  std::int32_t expert_id = -1;
  std::uint32_t file_id = 0;
  std::int64_t offset = 0;
  std::int64_t total_size = 0;
  std::vector<MemberMeta> members;
};

struct StageMeta {
  std::string name;
  bool is_sparse = false;
  std::uint32_t num_groups = 0;
};

struct StoreIndexV2 {
  std::string model_type;
  std::string checkpoint_name;
  std::uint64_t partition_size = 0;
  std::vector<StageMeta> stages;
  std::vector<GroupMeta> groups;

  const GroupMeta* FindExpertGroup(std::int32_t layer_id,
                                   std::int32_t expert_id) const;
  const GroupMeta* GroupOfTensor(std::uint32_t tensor_id) const;
  std::size_t NumTensors() const;
};

std::string DataFileName(std::uint32_t file_id);

// Parses <store_dir>/store_index; throws std::runtime_error on malformed
// input or invariant violations (G1/G2/G5).
StoreIndexV2 ReadIndex(const std::string& store_dir);

}  // namespace moe_store

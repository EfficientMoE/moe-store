// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "index_v2.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace moe_store {

namespace {

class Cursor {
 public:
  explicit Cursor(std::vector<char> data) : data_(std::move(data)) {}

  template <typename T>
  T Read() {
    T value;
    Require(sizeof(T));
    std::memcpy(&value, data_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return value;
  }

  std::string ReadString() {
    auto length = Read<std::uint16_t>();
    Require(length);
    std::string value(data_.data() + pos_, length);
    pos_ += length;
    return value;
  }

  void ReadRaw(void* dst, std::size_t n) {
    Require(n);
    std::memcpy(dst, data_.data() + pos_, n);
    pos_ += n;
  }

 private:
  void Require(std::size_t n) const {
    if (pos_ + n > data_.size()) {
      throw std::runtime_error("moe-store index: truncated file");
    }
  }

  std::vector<char> data_;
  std::size_t pos_ = 0;
};

void Validate(const StoreIndexV2& index) {
  std::uint32_t expected_tensor_id = 0;
  for (const auto& group : index.groups) {
    if (group.offset % kGroupAlignment != 0) {
      throw std::runtime_error("moe-store index: G2 group alignment");
    }
    if (group.offset + group.total_size >
        static_cast<std::int64_t>(index.partition_size)) {
      throw std::runtime_error("moe-store index: G1 partition straddle");
    }
    if (group.members.empty()) {
      throw std::runtime_error("moe-store index: empty group");
    }
    std::int64_t cursor = 0;
    for (const auto& member : group.members) {
      if (member.rel_offset % kMemberAlignment != 0) {
        throw std::runtime_error("moe-store index: G2 member alignment");
      }
      if (member.rel_offset < cursor) {
        throw std::runtime_error("moe-store index: member overlap");
      }
      cursor = member.rel_offset + member.size;
      if (member.tensor_id != expected_tensor_id++) {
        throw std::runtime_error("moe-store index: G5 tensor id order");
      }
    }
    if (cursor > group.total_size) {
      throw std::runtime_error("moe-store index: members exceed total_size");
    }
  }
}

}  // namespace

const GroupMeta* StoreIndexV2::FindExpertGroup(std::int32_t layer_id,
                                               std::int32_t expert_id) const {
  for (const auto& group : groups) {
    if (group.kind == GroupKind::kExpert && group.layer_id == layer_id &&
        group.expert_id == expert_id) {
      return &group;
    }
  }
  return nullptr;
}

const GroupMeta* StoreIndexV2::GroupOfTensor(std::uint32_t tensor_id) const {
  for (const auto& group : groups) {
    for (const auto& member : group.members) {
      if (member.tensor_id == tensor_id) return &group;
    }
  }
  return nullptr;
}

std::size_t StoreIndexV2::NumTensors() const {
  std::size_t count = 0;
  for (const auto& group : groups) count += group.members.size();
  return count;
}

std::string DataFileName(std::uint32_t file_id) {
  return std::string(kDataFilePrefix) + std::to_string(file_id);
}

StoreIndexV2 ReadIndex(const std::string& store_dir) {
  const std::string path = store_dir + "/" + kIndexFileName;
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("moe-store index: cannot open " + path);
  }
  std::vector<char> data(static_cast<std::size_t>(file.tellg()));
  file.seekg(0);
  file.read(data.data(), static_cast<std::streamsize>(data.size()));
  Cursor cursor(std::move(data));

  char magic[8];
  cursor.ReadRaw(magic, sizeof(magic));
  if (std::memcmp(magic, kIndexMagic, sizeof(magic)) != 0) {
    throw std::runtime_error("moe-store index: bad magic");
  }
  auto version = cursor.Read<std::uint32_t>();
  cursor.Read<std::uint32_t>();
  if (version != kIndexVersion) {
    throw std::runtime_error("moe-store index: unsupported version " +
                             std::to_string(version));
  }

  StoreIndexV2 index;
  index.partition_size = cursor.Read<std::uint64_t>();
  index.model_type = cursor.ReadString();
  index.checkpoint_name = cursor.ReadString();
  auto num_stages = cursor.Read<std::uint32_t>();
  auto num_groups = cursor.Read<std::uint64_t>();
  auto num_tensors = cursor.Read<std::uint64_t>();

  index.stages.reserve(num_stages);
  for (std::uint32_t i = 0; i < num_stages; ++i) {
    StageMeta stage;
    stage.name = cursor.ReadString();
    stage.is_sparse = cursor.Read<std::uint8_t>() != 0;
    stage.num_groups = cursor.Read<std::uint32_t>();
    index.stages.push_back(std::move(stage));
  }

  index.groups.reserve(num_groups);
  for (std::uint64_t g = 0; g < num_groups; ++g) {
    GroupMeta group;
    group.group_id = cursor.Read<std::uint64_t>();
    group.kind = static_cast<GroupKind>(cursor.Read<std::uint8_t>());
    group.layer_id = cursor.Read<std::int32_t>();
    group.expert_id = cursor.Read<std::int32_t>();
    group.file_id = cursor.Read<std::uint32_t>();
    group.offset = cursor.Read<std::int64_t>();
    group.total_size = cursor.Read<std::int64_t>();
    auto num_members = cursor.Read<std::uint32_t>();
    group.members.reserve(num_members);
    for (std::uint32_t m = 0; m < num_members; ++m) {
      MemberMeta member;
      member.tensor_id = cursor.Read<std::uint32_t>();
      member.name = cursor.ReadString();
      member.rel_offset = cursor.Read<std::int64_t>();
      member.size = cursor.Read<std::int64_t>();
      member.dtype = cursor.ReadString();
      auto ndim = cursor.Read<std::uint8_t>();
      member.shape.reserve(ndim);
      for (std::uint8_t d = 0; d < ndim; ++d) {
        member.shape.push_back(cursor.Read<std::int64_t>());
      }
      group.members.push_back(std::move(member));
    }
    index.groups.push_back(std::move(group));
  }

  if (index.NumTensors() != num_tensors) {
    throw std::runtime_error("moe-store index: tensor count mismatch");
  }
  Validate(index);
  return index;
}

}  // namespace moe_store

/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizer.cc
 * \brief Common validation and dispatch for the C++ tokenizer interface.
 */
#include <tokenizers_cpp.h>

namespace tokenizers {
namespace {

constexpr uint32_t kKnownEncodeFlags = static_cast<uint32_t>(EncodeFlags::kIgnoreSpecialTokens) |
                                       static_cast<uint32_t>(EncodeFlags::kIgnoreAddedTokens);

void ValidateEncodeOptions(const EncodeOptions& options, bool supports_encode_flags) {
  const uint32_t raw_flags = static_cast<uint32_t>(options.flags);
  if ((raw_flags & ~kKnownEncodeFlags) != 0) {
    throw TokenizerError(TokenizerErrorCode::kInvalidArgument,
                         "Encode options contain unknown flag bits");
  }
  if (raw_flags != 0 && !supports_encode_flags) {
    throw TokenizerError(TokenizerErrorCode::kUnsupportedOperation,
                         "Added-token ignore flags are unsupported by this tokenizer backend");
  }
}

}  // namespace

std::vector<int32_t> Tokenizer::Encode(const std::string& text) const {
  return Encode(text, EncodeOptions{});
}

std::vector<int32_t> Tokenizer::Encode(const std::string& text,
                                       const EncodeOptions& options) const {
  ValidateEncodeOptions(options, SupportsEncodeFlags());
  return EncodeImpl(text, options);
}

std::vector<std::vector<int32_t>> Tokenizer::EncodeBatch(
    const std::vector<std::string>& texts) const {
  return EncodeBatch(texts, EncodeOptions{});
}

std::vector<std::vector<int32_t>> Tokenizer::EncodeBatch(const std::vector<std::string>& texts,
                                                         const EncodeOptions& options) const {
  ValidateEncodeOptions(options, SupportsEncodeFlags());
  return EncodeBatchImpl(texts, options);
}

std::vector<std::vector<int32_t>> Tokenizer::EncodeBatchImpl(const std::vector<std::string>& texts,
                                                             const EncodeOptions& options) const {
  // Backends without a native batch implementation still receive the exact same per-call
  // options for every input.
  std::vector<std::vector<int32_t>> result;
  result.reserve(texts.size());
  for (const auto& text : texts) {
    result.push_back(EncodeImpl(text, options));
  }
  return result;
}

}  // namespace tokenizers

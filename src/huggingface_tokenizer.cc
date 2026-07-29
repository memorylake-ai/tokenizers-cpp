/*!
 *  Copyright (c) 2023 by Contributors
 * \file huggingface_tokenizer.cc
 * \brief Hugging Face tokenizer backed by the Rust tokenizers library.
 */
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

#include <cassert>
#include <limits>
#include <utility>

namespace tokenizers {
namespace {

// Owns one error message allocated by Rust. Copying the text before throwing keeps the C ABI
// allocation lifetime independent from the C++ exception.
class ErrorMessage {
 public:
  ErrorMessage() = default;
  ErrorMessage(const ErrorMessage&) = delete;
  ErrorMessage& operator=(const ErrorMessage&) = delete;

  ~ErrorMessage() { tokenizers_free_error_message(&message_); }

  ::TokenizerErrorMessage* out() { return &message_; }

  std::string str() const {
    if (message_.data == nullptr || message_.len == 0) {
      return "Tokenizer backend failed without an error message";
    }
    return std::string(message_.data, message_.len);
  }

 private:
  ::TokenizerErrorMessage message_{nullptr, 0};
};

[[noreturn]] void ThrowStatus(TokenizerStatus status, const ErrorMessage& error) {
  const std::string message = error.str();
  switch (status) {
    case TOKENIZERS_STATUS_INVALID_ARGUMENT:
      throw TokenizerError(TokenizerErrorCode::kInvalidArgument, message);
    case TOKENIZERS_STATUS_UNSUPPORTED:
      throw TokenizerError(TokenizerErrorCode::kUnsupportedOperation, message);
    case TOKENIZERS_STATUS_INTERNAL:
      throw TokenizerError(TokenizerErrorCode::kInternal, message);
    case TOKENIZERS_STATUS_OK:
      throw TokenizerError(TokenizerErrorCode::kInternal,
                           "Tokenizer reported success through its error path");
  }
  throw TokenizerError(TokenizerErrorCode::kInternal, "Tokenizer returned an unknown status");
}

void CheckStatus(TokenizerStatus status, const ErrorMessage& error) {
  if (status != TOKENIZERS_STATUS_OK) {
    ThrowStatus(status, error);
  }
}

// Owns every token-id buffer returned for one C ABI call, including partial results if a future
// backend starts reporting them on failure.
class EncodeResults {
 public:
  explicit EncodeResults(size_t count) : results_(count, TokenizerEncodeResult{nullptr, 0}) {}
  EncodeResults(const EncodeResults&) = delete;
  EncodeResults& operator=(const EncodeResults&) = delete;

  ~EncodeResults() { tokenizers_free_encode_results(results_.data(), results_.size()); }

  TokenizerEncodeResult* data() { return results_.data(); }
  const TokenizerEncodeResult& operator[](size_t index) const { return results_[index]; }

 private:
  std::vector<TokenizerEncodeResult> results_;
};

std::vector<int32_t> ConvertResult(const TokenizerEncodeResult& result) {
  if (result.len != 0 && result.token_ids == nullptr) {
    throw TokenizerError(TokenizerErrorCode::kInternal,
                         "Tokenizer returned a null token buffer with non-zero length");
  }

  std::vector<int32_t> converted;
  converted.reserve(result.len);
  for (size_t index = 0; index < result.len; ++index) {
    const uint32_t token_id = result.token_ids[index];
    if (token_id > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      throw TokenizerError(TokenizerErrorCode::kInternal,
                           "Tokenizer returned a token id that exceeds int32");
    }
    converted.push_back(static_cast<int32_t>(token_id));
  }
  return converted;
}

uint32_t RawFlags(EncodeFlags flags) { return static_cast<uint32_t>(flags); }

// Keeps a newly created Rust handle recoverable until ownership has been transferred into the
// C++ tokenizer object.
class HandleOwner {
 public:
  explicit HandleOwner(TokenizerHandle handle) : handle_(handle) {}
  HandleOwner(const HandleOwner&) = delete;
  HandleOwner& operator=(const HandleOwner&) = delete;

  ~HandleOwner() {
    if (handle_ != nullptr) {
      tokenizers_free(handle_);
    }
  }

  TokenizerHandle release() { return std::exchange(handle_, nullptr); }

 private:
  TokenizerHandle handle_;
};

}  // namespace

/*!
 * \brief C++ owner for one Rust tokenizer handle.
 *
 * JSON-backed instances expose immutable policy-specific encoding views and are safe for
 * concurrent Encode/EncodeBatch calls. Decode and token lookup retain mutable Rust-side scratch
 * strings and are not covered by that concurrency guarantee.
 */
class HFTokenizer : public Tokenizer {
 public:
  HFTokenizer(TokenizerHandle handle, bool supports_encode_flags)
      : handle_(handle), supports_encode_flags_(supports_encode_flags) {
#ifdef COMPILE_WASM_RUNTIME
    setenv("TOKENIZERS_PARALLELISM", "false", true);
#endif
  }

  HFTokenizer(const HFTokenizer&) = delete;
  HFTokenizer(HFTokenizer&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)),
        supports_encode_flags_(other.supports_encode_flags_) {}

  ~HFTokenizer() {
    if (handle_ != nullptr) {
      tokenizers_free(handle_);
    }
  }

  std::vector<int32_t> EncodeImpl(const std::string& text,
                                  const EncodeOptions& options) const final {
    EncodeResults results(1);
    ErrorMessage error;
    const TokenizerStatus status = tokenizers_encode(
        handle_, text.data(), text.length(), static_cast<int>(options.add_special_tokens),
        RawFlags(options.flags), results.data(), error.out());
    CheckStatus(status, error);
    return ConvertResult(results[0]);
  }

  std::vector<std::vector<int32_t>> EncodeBatchImpl(const std::vector<std::string>& texts,
                                                    const EncodeOptions& options) const final {
    if (texts.empty()) {
      return {};
    }

    std::vector<const char*> text_data;
    std::vector<size_t> text_lengths;
    text_data.reserve(texts.size());
    text_lengths.reserve(texts.size());
    for (const auto& text : texts) {
      text_data.push_back(text.data());
      text_lengths.push_back(text.length());
    }

    EncodeResults results(texts.size());
    ErrorMessage error;
    const TokenizerStatus status =
        tokenizers_encode_batch(handle_, text_data.data(), text_lengths.data(), texts.size(),
                                static_cast<int>(options.add_special_tokens),
                                RawFlags(options.flags), results.data(), error.out());
    CheckStatus(status, error);

    std::vector<std::vector<int32_t>> converted;
    converted.reserve(texts.size());
    for (size_t index = 0; index < texts.size(); ++index) {
      converted.push_back(ConvertResult(results[index]));
    }
    return converted;
  }

  std::string Decode(const std::vector<int32_t>& ids, bool skip_special_tokens) final {
    tokenizers_decode(handle_, reinterpret_cast<const uint32_t*>(ids.data()), ids.size(),
                      static_cast<int>(skip_special_tokens));
    const char* data;
    size_t len;
    tokenizers_get_decode_str(handle_, &data, &len);
    return std::string(data, len);
  }

  std::string Decode(const std::vector<int32_t>& ids) final { return Decode(ids, false); }

  size_t GetVocabSize() final {
    size_t size;
    tokenizers_get_vocab_size(handle_, &size);
    assert(size > 0);
    return size;
  }

  std::string IdToToken(int32_t id) final {
    const char* data;
    size_t len;
    tokenizers_id_to_token(handle_, static_cast<uint32_t>(id), &data, &len);
    return std::string(data, len);
  }

  int32_t TokenToId(const std::string& token) final {
    int32_t id;
    tokenizers_token_to_id(handle_, token.data(), token.length(), &id);
    return id;
  }

 protected:
  bool SupportsEncodeFlags() const noexcept final { return supports_encode_flags_; }

 private:
  TokenizerHandle handle_{nullptr};
  bool supports_encode_flags_;
};

std::unique_ptr<Tokenizer> Tokenizer::FromBlobJSON(const std::string& json) {
  TokenizerHandle handle = nullptr;
  ErrorMessage error;
  const TokenizerStatus status =
      tokenizers_new_from_str(json.data(), json.length(), &handle, error.out());
  CheckStatus(status, error);
  if (handle == nullptr) {
    throw TokenizerError(TokenizerErrorCode::kInternal,
                         "Tokenizer factory succeeded with a null handle");
  }
  HandleOwner handle_owner(handle);
  std::unique_ptr<HFTokenizer> tokenizer(new HFTokenizer(handle, true));
  handle_owner.release();
  return tokenizer;
}

std::unique_ptr<Tokenizer> Tokenizer::FromBlobByteLevelBPE(const std::string& vocab,
                                                           const std::string& merges,
                                                           const std::string& added_tokens) {
  TokenizerHandle handle = nullptr;
  ErrorMessage error;
  const TokenizerStatus status = byte_level_bpe_tokenizers_new_from_str(
      vocab.data(), vocab.length(), merges.data(), merges.length(), added_tokens.data(),
      added_tokens.length(), &handle, error.out());
  CheckStatus(status, error);
  if (handle == nullptr) {
    throw TokenizerError(TokenizerErrorCode::kInternal,
                         "BPE tokenizer factory succeeded with a null handle");
  }
  HandleOwner handle_owner(handle);
  std::unique_ptr<HFTokenizer> tokenizer(new HFTokenizer(handle, false));
  handle_owner.release();
  return tokenizer;
}

}  // namespace tokenizers

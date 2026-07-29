/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizers_cpp.h
 * \brief A C++ binding to common set of tokenizers
 */
#ifndef TOKENIZERS_CPP_H_
#define TOKENIZERS_CPP_H_

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tokenizers {

/*!
 * \brief Flags controlling how input added tokens participate in encoding.
 *
 * These flags affect only recognition of token text already present in the input. They are
 * independent from EncodeOptions::add_special_tokens, which controls post-processing.
 */
enum class EncodeFlags : uint32_t {
  kNone = 0,
  kIgnoreSpecialTokens = 1U << 0,
  kIgnoreAddedTokens = 1U << 1,
};

constexpr EncodeFlags operator|(EncodeFlags lhs, EncodeFlags rhs) {
  return static_cast<EncodeFlags>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

constexpr EncodeFlags operator&(EncodeFlags lhs, EncodeFlags rhs) {
  return static_cast<EncodeFlags>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

/*!
 * \brief Per-call encoding controls.
 *
 * The neutral defaults preserve the historical behavior: do not add post-processor tokens and
 * recognize all configured added tokens.
 */
struct EncodeOptions {
  bool add_special_tokens{false};
  EncodeFlags flags{EncodeFlags::kNone};
};

/*! \brief Stable categories for errors reported by the tokenizer API. */
enum class TokenizerErrorCode {
  kInvalidArgument,
  kUnsupportedOperation,
  kInternal,
};

/*!
 * \brief Exception carrying an inspectable tokenizer failure category.
 *
 * Encoding errors originating in Rust are converted to this type after the C ABI returns, so no
 * exception or Rust panic unwinds through the language boundary.
 */
class TokenizerError : public std::runtime_error {
 public:
  TokenizerError(TokenizerErrorCode code, const std::string& message)
      : std::runtime_error(message), code_(code) {}

  TokenizerErrorCode code() const noexcept { return code_; }

 private:
  TokenizerErrorCode code_;
};

/*!
 * \brief a universal tokenizer that loads
 *  either HF's tokenizer or sentence piece,
 *  depending on the constructor
 */
class Tokenizer {
 public:
  /*! \brief virtual destructor */
  virtual ~Tokenizer() {}

  /*!
   * \brief Encode text into ids.
   * \param text The input text.
   * \param options Per-call post-processing and added-token controls.
   * \returns The encoded token ids.
   * \throws TokenizerError for invalid flags, unsupported policies, or backend failures.
   */
  std::vector<int32_t> Encode(const std::string& text) const;
  std::vector<int32_t> Encode(const std::string& text, const EncodeOptions& options) const;

  /*!
   * \brief Encode a batch of texts into ids.
   * \param texts The input texts.
   * \param options Controls shared by every sequence in the batch.
   * \returns The encoded token ids.
   * \throws TokenizerError for invalid flags, unsupported policies, or backend failures.
   */
  std::vector<std::vector<int32_t>> EncodeBatch(const std::vector<std::string>& texts) const;
  std::vector<std::vector<int32_t>> EncodeBatch(const std::vector<std::string>& texts,
                                                const EncodeOptions& options) const;

  /*!
   * \brief Decode token ids into text.
   * \param text The token ids.
   * \returns The decoded text.
   */
  virtual std::string Decode(const std::vector<int32_t>& ids) = 0;

  /*!
   * \brief Decode token ids with explicit special-token handling.
   * \param ids The token ids.
   * \param skip_special_tokens Whether registered special tokens are omitted.
   * \returns The decoded text.
   */
  virtual std::string Decode(const std::vector<int32_t>& ids, bool skip_special_tokens) {
    (void)skip_special_tokens;
    return Decode(ids);
  }

  /*!
   * \brief Returns the vocabulary size. Special tokens are considered.
   */
  virtual size_t GetVocabSize() = 0;

  /*!
   * \brief Convert the given id to its corresponding token if it exists. If not, return an
   * empty string.
   */
  virtual std::string IdToToken(int32_t token_id) = 0;

  /*!
   * \brief Convert the given token to its corresponding id if it exists. If not, return -1.
   */
  virtual int32_t TokenToId(const std::string& token) = 0;

  //---------------------------------------------------
  // Factory functions from byte-blobs
  // These factory function takes in in-memory blobs
  // so the library can be independent from filesystem
  //---------------------------------------------------
  /*!
   * \brief Create HF tokenizer from a single in-memory json blob.
   *
   * \param json_blob The json blob.
   * \return The created tokenzier.
   */
  static std::unique_ptr<Tokenizer> FromBlobJSON(const std::string& json_blob);
  /*!
   * \brief Create BPE tokenizer
   *
   * \param vocab_blob The blob that contains vocabs.
   * \param merges_blob The blob that contains the merges.
   * \param added_tokens The added tokens.
   * \return The created tokenizer.
   */
  static std::unique_ptr<Tokenizer> FromBlobByteLevelBPE(const std::string& vocab_blob,
                                                         const std::string& merges_blob,
                                                         const std::string& added_tokens = "");
  /*!
   * \brief Create SentencePiece.
   *
   * \param model_blob The blob that contains vocabs.
   * \return The created tokenizer.
   */
  static std::unique_ptr<Tokenizer> FromBlobSentencePiece(const std::string& model_blob);
  /*!
   * \brief Create RWKVWorldTokenizer.
   *
   * \param model_blob The blob that contains vocabs.
   * \return The created tokenizer.
   */
  static std::unique_ptr<Tokenizer> FromBlobRWKVWorld(const std::string& model_blob);

 protected:
  /*!
   * \brief Backend-specific single-input implementation called after common validation.
   *
   * Implementations may assume the flags are known and supported by the backend.
   */
  virtual std::vector<int32_t> EncodeImpl(const std::string& text,
                                          const EncodeOptions& options) const = 0;

  /*!
   * \brief Backend-specific batch implementation called after common validation.
   *
   * The default preserves backend semantics by invoking EncodeImpl for every sequence.
   */
  virtual std::vector<std::vector<int32_t>> EncodeBatchImpl(const std::vector<std::string>& texts,
                                                            const EncodeOptions& options) const;

  /*! \brief Whether this backend implements non-default added-token ignore flags. */
  virtual bool SupportsEncodeFlags() const noexcept { return false; }
};

}  // namespace tokenizers
#endif  // TOKENIZERS_CPP_H_

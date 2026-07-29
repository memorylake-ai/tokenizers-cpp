/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizers_c.h
 * \brief C binding to tokenizers rust library
 */
#ifndef TOKENIZERS_C_H_
#define TOKENIZERS_C_H_

// The C API
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef void* TokenizerHandle;

/*! \brief Status returned by failable tokenizer C API calls. */
typedef enum {
  TOKENIZERS_STATUS_OK = 0,
  TOKENIZERS_STATUS_INVALID_ARGUMENT = 1,
  TOKENIZERS_STATUS_UNSUPPORTED = 2,
  TOKENIZERS_STATUS_INTERNAL = 3,
} TokenizerStatus;

typedef uint32_t TokenizerEncodeFlags;

/*! \brief Preserve recognition of every configured added token. */
#define TOKENIZERS_ENCODE_FLAG_NONE ((TokenizerEncodeFlags)0)
/*! \brief Encode added tokens marked special as ordinary input text. */
#define TOKENIZERS_ENCODE_FLAG_IGNORE_SPECIAL_TOKENS ((TokenizerEncodeFlags)1 << 0)
/*! \brief Encode every added token as ordinary input text. */
#define TOKENIZERS_ENCODE_FLAG_IGNORE_ADDED_TOKENS ((TokenizerEncodeFlags)1 << 1)

/*! \brief Rust-owned token IDs returned by an encode call. */
typedef struct {
  uint32_t* token_ids;
  size_t len;
} TokenizerEncodeResult;

/*! \brief Rust-owned error text returned when a failable call does not succeed. */
typedef struct {
  char* data;
  size_t len;
} TokenizerErrorMessage;

/*!
 * \brief Construct a Hugging Face JSON tokenizer.
 *
 * On success, ownership of out_handle passes to the caller. On failure, out_error must be released
 * with tokenizers_free_error_message.
 */
TokenizerStatus tokenizers_new_from_str(const char* json, size_t len, TokenizerHandle* out_handle,
                                        TokenizerErrorMessage* out_error);

TokenizerStatus byte_level_bpe_tokenizers_new_from_str(const char* vocab, size_t vocab_len,
                                                       const char* merges, size_t merges_len,
                                                       const char* added_tokens,
                                                       size_t added_tokens_len,
                                                       TokenizerHandle* out_handle,
                                                       TokenizerErrorMessage* out_error);

/*!
 * \brief Encode one UTF-8 string with per-call post-processing and added-token controls.
 *
 * The caller releases a successful out_result with tokenizers_free_encode_results and a failed
 * out_error with tokenizers_free_error_message.
 */
TokenizerStatus tokenizers_encode(TokenizerHandle handle, const char* data, size_t len,
                                  int add_special_tokens, TokenizerEncodeFlags flags,
                                  TokenizerEncodeResult* out_result,
                                  TokenizerErrorMessage* out_error);

/*! \brief Encode a batch using one common set of flags for every sequence. */
TokenizerStatus tokenizers_encode_batch(TokenizerHandle handle, const char* const* data,
                                        const size_t* len, size_t num_seqs, int add_special_tokens,
                                        TokenizerEncodeFlags flags,
                                        TokenizerEncodeResult* out_results,
                                        TokenizerErrorMessage* out_error);

void tokenizers_free_encode_results(TokenizerEncodeResult* results, size_t num_seqs);

void tokenizers_free_error_message(TokenizerErrorMessage* error);

void tokenizers_decode(TokenizerHandle handle, const uint32_t* data, size_t len,
                       int skip_special_token);

void tokenizers_get_decode_str(TokenizerHandle handle, const char** data, size_t* len);

void tokenizers_get_vocab_size(TokenizerHandle handle, size_t* size);

void tokenizers_id_to_token(TokenizerHandle handle, uint32_t id, const char** data, size_t* len);

// tokenizers_token_to_id stores -1 to *id if the token is not in the vocab
void tokenizers_token_to_id(TokenizerHandle handle, const char* token, size_t len, int32_t* id);

void tokenizers_free(TokenizerHandle handle);

#ifdef __cplusplus
}
#endif
#endif  // TOKENIZERS_C_H_

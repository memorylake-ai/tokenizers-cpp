#include <tokenizers_cpp.h>

#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using tokenizers::EncodeFlags;
using tokenizers::EncodeOptions;
using tokenizers::Tokenizer;
using tokenizers::TokenizerError;
using tokenizers::TokenizerErrorCode;

constexpr const char* kTokenizerJson = R"json(
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 11,
      "content": "<special>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 12,
      "content": "<added>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    }
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": false
  },
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<special>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}}
    ],
    "pair": [
      {"Sequence": {"id": "A", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}}
    ],
    "special_tokens": {
      "<special>": {
        "id": "<special>",
        "ids": [11],
        "tokens": ["<special>"]
      }
    }
  },
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "[UNK]",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {
      "[UNK]": 0,
      "<": 1,
      ">": 2,
      "s": 3,
      "p": 4,
      "e": 5,
      "c": 6,
      "i": 7,
      "a": 8,
      "l": 9,
      "d": 10
    },
    "merges": []
  }
}
)json";

const std::vector<int32_t> kOrdinarySpecial = {1, 3, 4, 5, 6, 7, 8, 9, 2};
const std::vector<int32_t> kOrdinaryAdded = {1, 8, 10, 10, 5, 10, 2};

template <typename T>
void ExpectEqual(const T& actual, const T& expected, const std::string& context) {
  if (actual != expected) {
    throw std::runtime_error("Unexpected result: " + context);
  }
}

template <typename Function>
void ExpectTokenizerError(Function&& function, TokenizerErrorCode expected_code,
                          const std::string& context) {
  try {
    function();
  } catch (const TokenizerError& error) {
    if (error.code() != expected_code) {
      throw std::runtime_error("Unexpected tokenizer error category: " + context);
    }
    if (std::string(error.what()).empty()) {
      throw std::runtime_error("Tokenizer error message is empty: " + context);
    }
    return;
  }
  throw std::runtime_error("Expected tokenizer error was not raised: " + context);
}

EncodeOptions Options(EncodeFlags flags, bool add_special_tokens = false) {
  return EncodeOptions{add_special_tokens, flags};
}

std::vector<int32_t> Concatenate(const std::vector<int32_t>& lhs, const std::vector<int32_t>& rhs) {
  std::vector<int32_t> result = lhs;
  result.insert(result.end(), rhs.begin(), rhs.end());
  return result;
}

void TestFlagSemantics() {
  std::unique_ptr<Tokenizer> tokenizer = Tokenizer::FromBlobJSON(kTokenizerJson);

  ExpectEqual(tokenizer->Encode("<special>"), std::vector<int32_t>{11}, "default special");
  ExpectEqual(tokenizer->Encode("<added>"), std::vector<int32_t>{12}, "default added");

  const EncodeOptions ignore_special = Options(EncodeFlags::kIgnoreSpecialTokens);
  ExpectEqual(tokenizer->Encode("<special>", ignore_special), kOrdinarySpecial, "ignore special");
  ExpectEqual(tokenizer->Encode("<added>", ignore_special), std::vector<int32_t>{12},
              "ignore special keeps non-special added token");
  ExpectEqual(tokenizer->Encode("<special><added>", ignore_special),
              Concatenate(kOrdinarySpecial, {12}), "mixed special-only policy");

  const EncodeOptions ignore_added = Options(EncodeFlags::kIgnoreAddedTokens);
  ExpectEqual(tokenizer->Encode("<special>", ignore_added), kOrdinarySpecial,
              "ignore all added handles special token");
  ExpectEqual(tokenizer->Encode("<added>", ignore_added), kOrdinaryAdded,
              "ignore all added handles non-special token");
  const std::vector<int32_t> ordinary_mixed = Concatenate(kOrdinarySpecial, kOrdinaryAdded);
  ExpectEqual(tokenizer->Encode("<special><added>", ignore_added), ordinary_mixed,
              "mixed all-added policy");
  ExpectEqual(tokenizer->Decode(ordinary_mixed), std::string("<special><added>"),
              "ordinary tokens round trip");

  const EncodeOptions both =
      Options(EncodeFlags::kIgnoreSpecialTokens | EncodeFlags::kIgnoreAddedTokens);
  ExpectEqual(tokenizer->Encode("<special><added>", both), ordinary_mixed,
              "combined flags equal all-added policy");

  const std::vector<int32_t> post_processed =
      tokenizer->Encode("<added>", Options(EncodeFlags::kIgnoreAddedTokens, true));
  ExpectEqual(post_processed, Concatenate({11}, kOrdinaryAdded),
              "post-processor and ignore policy are orthogonal");

  ExpectEqual(tokenizer->Encode("", ignore_added), std::vector<int32_t>{}, "empty single input");
  ExpectEqual(tokenizer->EncodeBatch({}, ignore_added), std::vector<std::vector<int32_t>>{},
              "empty batch");
}

void TestBatchParity() {
  std::unique_ptr<Tokenizer> tokenizer = Tokenizer::FromBlobJSON(kTokenizerJson);
  const std::vector<std::string> texts = {"<special><added>", "", "<added><added>"};
  for (const EncodeFlags flags :
       {EncodeFlags::kNone, EncodeFlags::kIgnoreSpecialTokens, EncodeFlags::kIgnoreAddedTokens}) {
    const EncodeOptions options = Options(flags);
    std::vector<std::vector<int32_t>> expected;
    expected.reserve(texts.size());
    for (const auto& text : texts) {
      expected.push_back(tokenizer->Encode(text, options));
    }
    ExpectEqual(tokenizer->EncodeBatch(texts, options), expected, "batch/single parity");
  }
}

void TestErrorsAndStateIsolation() {
  std::unique_ptr<Tokenizer> tokenizer = Tokenizer::FromBlobJSON(kTokenizerJson);

  const EncodeFlags unknown_flag = static_cast<EncodeFlags>(1U << 31);
  ExpectTokenizerError(
      [&] { static_cast<void>(tokenizer->Encode("<special>", Options(unknown_flag))); },
      TokenizerErrorCode::kInvalidArgument, "unknown C++ flag");
  ExpectEqual(tokenizer->Encode("<special>"), std::vector<int32_t>{11},
              "default behavior after invalid flag");

  const std::string invalid_utf8(1, static_cast<char>(0xff));
  ExpectTokenizerError([&] { static_cast<void>(tokenizer->Encode(invalid_utf8)); },
                       TokenizerErrorCode::kInvalidArgument, "Rust-to-C++ invalid UTF-8 error");
  ExpectEqual(tokenizer->Encode("<added>"), std::vector<int32_t>{12},
              "default behavior after Rust error");

  ExpectTokenizerError([] { static_cast<void>(Tokenizer::FromBlobJSON("{")); },
                       TokenizerErrorCode::kInvalidArgument, "invalid tokenizer JSON");

  std::unique_ptr<Tokenizer> byte_bpe =
      Tokenizer::FromBlobByteLevelBPE(R"({"[UNK]": 0, "a": 1})", "");
  ExpectEqual(byte_bpe->Encode("a"), std::vector<int32_t>{1}, "default byte BPE encoding");
  ExpectEqual(byte_bpe->EncodeBatch({"a", "a"}), std::vector<std::vector<int32_t>>{{1}, {1}},
              "default byte BPE batch encoding");
  ExpectTokenizerError(
      [&] { static_cast<void>(byte_bpe->Encode("a", Options(EncodeFlags::kIgnoreSpecialTokens))); },
      TokenizerErrorCode::kUnsupportedOperation, "non-JSON backend rejects ignore flags");
  ExpectTokenizerError(
      [&] {
        static_cast<void>(byte_bpe->EncodeBatch({"a"}, Options(EncodeFlags::kIgnoreAddedTokens)));
      },
      TokenizerErrorCode::kUnsupportedOperation, "non-JSON backend rejects batch ignore flags");
}

void TestConcurrentPolicies() {
  std::unique_ptr<Tokenizer> tokenizer = Tokenizer::FromBlobJSON(kTokenizerJson);
  const Tokenizer* shared_tokenizer = tokenizer.get();
  const std::vector<int32_t> ordinary_mixed = Concatenate(kOrdinarySpecial, kOrdinaryAdded);
  const std::vector<std::pair<EncodeFlags, std::vector<int32_t>>> policies = {
      {EncodeFlags::kNone, {11, 12}},
      {EncodeFlags::kIgnoreSpecialTokens, Concatenate(kOrdinarySpecial, {12})},
      {EncodeFlags::kIgnoreAddedTokens, ordinary_mixed},
  };

  std::promise<void> start;
  std::shared_future<void> gate = start.get_future().share();
  std::vector<std::future<void>> workers;
  workers.reserve(policies.size());
  for (const auto& [flags, expected] : policies) {
    workers.push_back(std::async(std::launch::async, [=] {
      gate.wait();
      const EncodeOptions options = Options(flags);
      for (size_t iteration = 0; iteration < 100; ++iteration) {
        ExpectEqual(shared_tokenizer->Encode("<special><added>", options), expected,
                    "concurrent single encode");
        ExpectEqual(
            shared_tokenizer->EncodeBatch({"<special><added>", "<special><added>"}, options),
            std::vector<std::vector<int32_t>>{expected, expected}, "concurrent batch encode");
      }
    }));
  }

  start.set_value();
  for (auto& worker : workers) {
    worker.get();
  }
  ExpectEqual(tokenizer->Encode("<special><added>"), std::vector<int32_t>({11, 12}),
              "default behavior after concurrent policies");
}

}  // namespace

int main() {
  try {
    TestFlagSemantics();
    TestBatchParity();
    TestErrorsAndStateIsolation();
    TestConcurrentPolicies();
  } catch (const std::exception& error) {
    std::cerr << "tokenizers_cpp_tests failed: " << error.what() << std::endl;
    return 1;
  }

  std::cout << "tokenizers_cpp_tests passed" << std::endl;
  return 0;
}

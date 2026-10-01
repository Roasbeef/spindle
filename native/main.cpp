// Spindle keeps llama.cpp in an external process. The reader remains live
// during model load and inference, so stdin EOF or shutdown ends all native
// work when the reader receives a complete shutdown frame or detects EOF.
// The BEAM owner observes process exit before reporting confirmed drain.
//
// Flow: main starts receive before model loading, constructs Engine, and sends
// the validated greeting. receive bounds frames before allocating their bodies
// and transfers one pending request through Inbox. main moves that request out
// under the mutex and calls serve after releasing the lock. serve validates
// every text through tokenize before embed evaluates inputs sequentially.
// send publishes one complete response; failure publishes one terminal error.
//
// receive handles shutdown and owner loss independently of that request path.
// It calls _Exit, so the OS reclaims model memory and threads even while main
// is loading, evaluating, or writing stdout. Port closure requests this path;
// only the BEAM exit-status message confirms it ran to native process exit.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Bytes = std::vector<uint8_t>;
constexpr uint32_t max_frame = 1024 * 1024;
constexpr uint32_t max_batch = 16;
constexpr uint32_t max_text = 65536;
constexpr uint32_t context_limit = 2048;

// Reader owns a cursor into one bounded frame. Length checks precede string
// construction, so malformed input cannot allocate from an unchecked length.
struct Reader {
    // The caller retains the complete frame for this reader's lifetime.
    const Bytes & bytes;

    // Every successful read advances within bytes; no unchecked offset escapes.
    size_t pos = 0;

    // u32 advances the cursor only when a complete integer remains.
    uint32_t u32() {
        if (bytes.size() - pos < 4) throw std::runtime_error("truncated integer");
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) value = (value << 8) | bytes[pos++];
        return value;
    }

    // text validates both the per-input limit and the remaining frame.
    std::string text() {
        const uint32_t size = u32();
        if (size > max_text || size > bytes.size() - pos)
            throw std::runtime_error("invalid text length");
        std::string value(reinterpret_cast<const char *>(bytes.data() + pos), size);
        pos += size;
        return value;
    }

    // end rejects trailing data instead of accepting a valid prefix.
    void end() {
        if (pos != bytes.size()) throw std::runtime_error("trailing request bytes");
    }
};

// Inbox transfers one pending frame from the stdin reader to model execution.
// Shutdown bypasses this slot, so a busy model cannot delay owner teardown.
struct Inbox {
    // Protects pending during the reader-to-main ownership transfer.
    std::mutex mutex;

    // Wakes main only after pending contains a complete bounded frame.
    std::condition_variable ready;

    // Empty means no queued work. There can be one active request in main and
    // one pending frame here; public Gleam calls never pipeline requests.
    Bytes pending;
};

// Engine owns the llama.cpp model and context on the inference thread. Normal
// exception unwinding releases both; immediate shutdown uses OS reclamation.
struct Engine {
    // Loading installs the model before a context can refer to it.
    llama_model * model = nullptr;

    // Main alone creates, evaluates, and frees this context.
    llama_context * context = nullptr;

    // Free the context before the model it references on ordinary unwinding.
    // _Exit deliberately bypasses destructors and relies on OS reclamation.
    ~Engine() {
        if (context) llama_free(context);
        if (model) llama_model_free(model);
    }
};

// u32 emits the protocol byte order independently of host endianness.
void u32(Bytes & out, uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) out.push_back(value >> shift);
}

// write_all handles short writes without allowing a partial frame to become
// a successful response. A broken output channel ends the helper.
void write_all(const uint8_t * bytes, size_t size) {
    while (size > 0) {
        const auto written = std::fwrite(bytes, 1, size, stdout);
        if (!written) std::_Exit(74);
        bytes += written;
        size -= written;
    }
}

// send publishes one complete bounded response. Only the inference thread
// writes stdout; diagnostics stay on stderr and cannot corrupt framing.
void send(const Bytes & body) {
    if (body.empty() || body.size() > max_frame) std::_Exit(70);
    Bytes header;
    u32(header, static_cast<uint32_t>(body.size()));
    write_all(header.data(), header.size());
    write_all(body.data(), body.size());
    if (std::fflush(stdout)) std::_Exit(74);
}

// failure reports one terminal request error, with no partial vector batch.
// The text bound also applies to exception messages from the native runtime.
void failure(uint32_t id, const std::string & message) {
    Bytes response{255};
    u32(response, id);
    const auto text = message.substr(0, 4096);
    u32(response, static_cast<uint32_t>(text.size()));
    response.insert(response.end(), text.begin(), text.end());
    send(response);
}

// EOF is loss of the owner. Do not wait for a model callback, which may
// not be invoked during GPU execution or model loading.
void read_all(uint8_t * bytes, size_t size) {
    while (size > 0) {
        const auto received = std::fread(bytes, 1, size, stdin);
        if (!received) std::_Exit(std::feof(stdin) ? 0 : 74);
        bytes += received;
        size -= received;
    }
}

// receive is the independent cancellation path. It reads bounded frames while
// the main thread loads or evaluates the model, and terminates the entire
// process on shutdown or EOF rather than waiting for model cooperation.
void receive(const std::shared_ptr<Inbox> & inbox) {
    for (;;) {
        Bytes header(4);
        read_all(header.data(), header.size());
        Reader parser{header};
        const uint32_t size = parser.u32();
        if (size == 0 || size > max_frame) std::_Exit(65);
        Bytes body(size);
        read_all(body.data(), size);

        // Shutdown has no model dependency and bypasses the work queue.
        if (body[0] == 3 && body.size() == 5) std::_Exit(0);

        // Publication transfers the frame, not model custody. A second queued
        // frame is a protocol violation, so there is no growing native queue.
        std::lock_guard<std::mutex> lock(inbox->mutex);
        if (!inbox->pending.empty()) std::_Exit(65);
        inbox->pending = std::move(body);
        inbox->ready.notify_one();
    }
}

// tokenize counts before allocating token storage and refuses truncation.
// Counting and embedding share this function, including special-token policy.
std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    // Both operations use identical special-token handling. User text is
    // ordinary text; strings resembling control tokens are not interpreted.
    int32_t count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                  nullptr, 0, true, false);
    if (count == INT32_MIN) throw std::runtime_error("tokenization failed");
    count = count < 0 ? -count : count;
    if (count == 0 || count > static_cast<int32_t>(context_limit))
        throw std::runtime_error("input exceeds the token limit or is empty");
    std::vector<llama_token> tokens(count);

    // The size-only pass bounded storage before the second pass fills it.
    const int32_t actual = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                         tokens.data(), count, true, false);
    if (actual != count) throw std::runtime_error("tokenizer length changed");
    return tokens;
}

// embed clears prior sequence state before each input and normalizes the
// selected pooled output. The returned vector cannot depend on a previous
// batch through retained KV state.
std::vector<float> embed(Engine & engine, const std::vector<llama_token> & tokens, uint32_t dims) {
    const auto memory = llama_get_memory(engine.context);
    if (memory) llama_memory_clear(memory, true);

    // One sequence occupies the context. Batch storage is freed after the
    // encoder/decoder returns, before output validation can throw.
    auto batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    const auto status = llama_model_has_encoder(engine.model)
        ? llama_encode(engine.context, batch) : llama_decode(engine.context, batch);
    llama_batch_free(batch);
    if (status != 0) throw std::runtime_error("inference failed");

    // The pooled output belongs to the context. Copy normalized components
    // before the next input clears sequence memory and overwrites that output.
    const auto values = llama_get_embeddings_seq(engine.context, 0);
    if (!values) throw std::runtime_error("model returned no pooled embedding");
    double squared = 0;
    for (uint32_t i = 0; i < dims; ++i) {
        if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite embedding");
        squared += static_cast<double>(values[i]) * values[i];
    }
    if (!(squared > 0) || !std::isfinite(squared))
        throw std::runtime_error("invalid embedding norm");

    // A failed norm check returns no vector; normalization never hides an
    // invalid model output by dividing through zero or a nonfinite value.
    std::vector<float> normalized(dims);
    for (uint32_t i = 0; i < dims; ++i)
        normalized[i] = static_cast<float>(values[i] / std::sqrt(squared));
    return normalized;
}

// serve validates every input before performing any inference. Responses are
// assembled privately and sent only after the entire operation succeeds.
void serve(const Bytes & request, Engine & engine, uint32_t dims) {
    Reader reader{request};
    const auto command = request[reader.pos++];
    uint32_t id = 0;
    try {
        id = reader.u32();
        if (command != 1 && command != 2) throw std::runtime_error("unknown command");
        const auto count = reader.u32();
        if (count == 0 || count > max_batch) throw std::runtime_error("invalid batch size");

        // Materialize all tokenized inputs before inference. An invalid later
        // text therefore cannot produce a partially successful wire batch.
        std::vector<std::vector<llama_token>> inputs;
        for (uint32_t i = 0; i < count; ++i)
            inputs.push_back(tokenize(llama_model_get_vocab(engine.model), reader.text()));
        reader.end();

        // Build the whole response privately. Main is the only stdout writer,
        // so the reader can interrupt without interleaving a protocol frame.
        Bytes response{static_cast<uint8_t>(command | 128)};
        u32(response, id);
        u32(response, count);
        if (command == 1) u32(response, dims);
        for (const auto & input : inputs) {
            if (command == 2) {
                u32(response, static_cast<uint32_t>(input.size()));
                continue;
            }
            for (const auto value : embed(engine, input, dims)) {
                // memcpy preserves IEEE float bits without aliasing through a
                // uint32_t pointer; u32 emits the network byte order.
                uint32_t bits;
                static_assert(sizeof(bits) == sizeof(value));
                std::memcpy(&bits, &value, sizeof(bits));
                u32(response, bits);
            }
        }
        send(response);
    } catch (const std::exception & error) {
        failure(id, error.what());
    }
}

// main starts the cancellation reader before any model work, then retains one
// loaded model across sequential requests. The greeting publishes its bounds
// only after context construction and pooling validation succeed.
int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: spindle-helper MODEL.gguf\n");
        return 64;
    }

    // Starting the reader first makes EOF and shutdown independent of model
    // loading. Shared ownership keeps Inbox alive in the detached thread.
    auto inbox = std::make_shared<Inbox>();
    std::thread(receive, inbox).detach();
    try {
        llama_backend_init();
        Engine engine;
        auto model_options = llama_model_default_params();
        model_options.n_gpu_layers = 0;
        engine.model = llama_model_load_from_file(argv[1], model_options);
        if (!engine.model) throw std::runtime_error("model load failed");

        // The initial profile is CPU-only with one bounded sequence. Merely
        // compiling a GPU backend does not enable offload in these options.
        auto options = llama_context_default_params();
        options.n_ctx = context_limit;
        options.n_batch = context_limit;
        options.n_ubatch = context_limit;
        options.n_seq_max = 1;
        options.n_threads = 4;
        options.n_threads_batch = 4;
        options.embeddings = true;
        engine.context = llama_init_from_model(engine.model, options);
        if (!engine.context) throw std::runtime_error("context initialization failed");

        // Publish startup only after the context can supply a supported pooled
        // embedding shape. Dimensions describe output, not model identity.
        const auto pooling = llama_pooling_type(engine.context);
        if (pooling != LLAMA_POOLING_TYPE_MEAN && pooling != LLAMA_POOLING_TYPE_CLS &&
            pooling != LLAMA_POOLING_TYPE_LAST) throw std::runtime_error("unsupported pooling");
        const auto dims = llama_model_n_embd_out(engine.model);
        if (dims <= 0 || dims > 4096) throw std::runtime_error("unsupported dimensions");
        Bytes ready{0, 0, 1};
        u32(ready, static_cast<uint32_t>(dims));
        u32(ready, context_limit);
        send(ready);

        // Move queued bytes out before model work and release the mutex. The
        // reader stays runnable throughout inference and stdout publication.
        for (;;) {
            Bytes request;
            {
                std::unique_lock<std::mutex> lock(inbox->mutex);
                inbox->ready.wait(lock, [&] { return !inbox->pending.empty(); });
                request = std::move(inbox->pending);
                inbox->pending.clear();
            }
            serve(request, engine, static_cast<uint32_t>(dims));
        }
    } catch (const std::exception & error) {
        // ID zero distinguishes startup failure from a correlated request
        // error. The helper exits instead of accepting work without a model.
        failure(0, error.what());
        std::_Exit(70);
    }
}

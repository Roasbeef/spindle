// Spindle keeps llama.cpp in an external process. The reader remains live
// during model load and inference, so stdin EOF or shutdown ends all native
// work immediately. The owner observes process exit before reporting drain.
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

struct Reader {
    const Bytes & bytes;
    size_t pos = 0;
    uint32_t u32() {
        if (bytes.size() - pos < 4) throw std::runtime_error("truncated integer");
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) value = (value << 8) | bytes[pos++];
        return value;
    }
    std::string text() {
        const uint32_t size = u32();
        if (size > max_text || size > bytes.size() - pos)
            throw std::runtime_error("invalid text length");
        std::string value(reinterpret_cast<const char *>(bytes.data() + pos), size);
        pos += size;
        return value;
    }
    void end() {
        if (pos != bytes.size()) throw std::runtime_error("trailing request bytes");
    }
};

void u32(Bytes & out, uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) out.push_back(value >> shift);
}

void write_all(const uint8_t * bytes, size_t size) {
    while (size > 0) {
        const auto written = std::fwrite(bytes, 1, size, stdout);
        if (!written) std::_Exit(74);
        bytes += written;
        size -= written;
    }
}

void send(const Bytes & body) {
    if (body.empty() || body.size() > max_frame) std::_Exit(70);
    Bytes header;
    u32(header, static_cast<uint32_t>(body.size()));
    write_all(header.data(), header.size());
    write_all(body.data(), body.size());
    if (std::fflush(stdout)) std::_Exit(74);
}

void failure(uint32_t id, const std::string & message) {
    Bytes response{255};
    u32(response, id);
    const auto text = message.substr(0, 4096);
    u32(response, static_cast<uint32_t>(text.size()));
    response.insert(response.end(), text.begin(), text.end());
    send(response);
}

struct Inbox {
    std::mutex mutex;
    std::condition_variable ready;
    Bytes pending;
};

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
        std::lock_guard<std::mutex> lock(inbox->mutex);
        if (!inbox->pending.empty()) std::_Exit(65);
        inbox->pending = std::move(body);
        inbox->ready.notify_one();
    }
}

struct Engine {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    ~Engine() {
        if (context) llama_free(context);
        if (model) llama_model_free(model);
    }
};

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
    const int32_t actual = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                         tokens.data(), count, true, false);
    if (actual != count) throw std::runtime_error("tokenizer length changed");
    return tokens;
}

std::vector<float> embed(Engine & engine, const std::vector<llama_token> & tokens, uint32_t dims) {
    const auto memory = llama_get_memory(engine.context);
    if (memory) llama_memory_clear(memory, true);
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
    const auto values = llama_get_embeddings_seq(engine.context, 0);
    if (!values) throw std::runtime_error("model returned no pooled embedding");
    double squared = 0;
    for (uint32_t i = 0; i < dims; ++i) {
        if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite embedding");
        squared += static_cast<double>(values[i]) * values[i];
    }
    if (!(squared > 0) || !std::isfinite(squared))
        throw std::runtime_error("invalid embedding norm");
    std::vector<float> normalized(dims);
    for (uint32_t i = 0; i < dims; ++i)
        normalized[i] = static_cast<float>(values[i] / std::sqrt(squared));
    return normalized;
}

void serve(const Bytes & request, Engine & engine, uint32_t dims) {
    Reader reader{request};
    const auto command = request[reader.pos++];
    uint32_t id = 0;
    try {
        id = reader.u32();
        if (command != 1 && command != 2) throw std::runtime_error("unknown command");
        const auto count = reader.u32();
        if (count == 0 || count > max_batch) throw std::runtime_error("invalid batch size");
        std::vector<std::vector<llama_token>> inputs;
        for (uint32_t i = 0; i < count; ++i)
            inputs.push_back(tokenize(llama_model_get_vocab(engine.model), reader.text()));
        reader.end();
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

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: spindle-helper MODEL.gguf\n");
        return 64;
    }
    auto inbox = std::make_shared<Inbox>();
    std::thread(receive, inbox).detach();
    try {
        llama_backend_init();
        Engine engine;
        auto model_options = llama_model_default_params();
        model_options.n_gpu_layers = 0;
        engine.model = llama_model_load_from_file(argv[1], model_options);
        if (!engine.model) throw std::runtime_error("model load failed");
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
        const auto pooling = llama_pooling_type(engine.context);
        if (pooling != LLAMA_POOLING_TYPE_MEAN && pooling != LLAMA_POOLING_TYPE_CLS &&
            pooling != LLAMA_POOLING_TYPE_LAST) throw std::runtime_error("unsupported pooling");
        const auto dims = llama_model_n_embd_out(engine.model);
        if (dims <= 0 || dims > 4096) throw std::runtime_error("unsupported dimensions");
        Bytes ready{0, 0, 1};
        u32(ready, static_cast<uint32_t>(dims));
        u32(ready, context_limit);
        send(ready);
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
        failure(0, error.what());
        std::_Exit(70);
    }
}

const std = @import("std");
const chilli = @import("chilli");
const zigformer = @import("zigformer");
const llm = zigformer.llm;
const vocab = zigformer.vocab;

const index_html = @embedFile("gui/index.html");

const Config = struct {
    port: u16 = 8085,
    host: []const u8 = "0.0.0.0",
    pretrain_path: []const u8 = "datasets/simple_dataset/pretrain.json",
    train_path: []const u8 = "datasets/simple_dataset/train.json",
    load_model_path: ?[]const u8 = null,
    max_request_size: usize = 1024 * 1024, // 1MB
    timeout_seconds: u32 = 30,
    max_prompt_length: usize = 1000,
    owns_memory: bool = false,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        if (self.owns_memory) {
            allocator.free(self.host);
            allocator.free(self.pretrain_path);
            allocator.free(self.train_path);
            if (self.load_model_path) |p| allocator.free(p);
        }
    }
};

// Global state holders for the server
const ServerState = struct {
    allocator: std.mem.Allocator,
    model: *llm.LLM,
    mutex: std.Thread.Mutex,
};

fn splitText(text: []const u8, allocator: std.mem.Allocator) !std.ArrayList([]const u8) {
    var list = std.ArrayList([]const u8){};
    errdefer list.deinit(allocator);
    var it = std.mem.splitScalar(u8, text, ' ');
    while (it.next()) |word| {
        try list.append(allocator, word);
    }
    return list;
}

fn readJsonLines(allocator: std.mem.Allocator, path: []const u8) !std.json.Parsed([]const []const u8) {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const contents = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(contents);

    return std.json.parseFromSlice([]const []const u8, allocator, contents, .{});
}

fn buildVocabFromDatasets(allocator: std.mem.Allocator, pretrain: []const []const u8, chat: []const []const u8) !vocab.Vocab {
    var vocab_set = std.StringHashMap(void).init(allocator);
    defer vocab_set.deinit();

    try vocab_set.put("</s>", {});

    for (pretrain) |text| {
        var words = try splitText(text, allocator);
        defer words.deinit(allocator);
        for (words.items) |word| {
            try vocab_set.put(word, {});
        }
    }

    for (chat) |text| {
        var words = try splitText(text, allocator);
        defer words.deinit(allocator);
        for (words.items) |word| {
            try vocab_set.put(word, {});
        }
    }

    var vocab_words = std.ArrayList([]const u8){};
    defer vocab_words.deinit(allocator);
    var it = vocab_set.keyIterator();
    while (it.next()) |key| {
        try vocab_words.append(allocator, key.*);
    }

    // Sort for deterministic vocab
    std.sort.pdq([]const u8, vocab_words.items, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);

    var v = vocab.Vocab.init(allocator);
    errdefer v.deinit();
    try v.build(vocab_words.items);
    return v;
}

fn loadConfig(allocator: std.mem.Allocator, path: []const u8) !Config {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        std.debug.print("Error: Could not open config file '{s}': {}\n", .{ path, err });
        return err;
    };
    defer file.close();

    const contents = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch |err| {
        std.debug.print("Error: Could not read config file: {}\n", .{err});
        return err;
    };
    defer allocator.free(contents);

    const parsed = std.json.parseFromSlice(Config, allocator, contents, .{ .ignore_unknown_fields = true }) catch |err| {
        std.debug.print("Error: Invalid JSON in config file: {}\n", .{err});
        return err;
    };
    defer parsed.deinit();

    // Duplicate strings because parsed.deinit() frees them
    var config = parsed.value;
    config.host = try allocator.dupe(u8, config.host);
    config.pretrain_path = try allocator.dupe(u8, config.pretrain_path);
    config.train_path = try allocator.dupe(u8, config.train_path);
    if (config.load_model_path) |p| config.load_model_path = try allocator.dupe(u8, p);
    config.owns_memory = true;

    return config;
}

fn validateConfig(config: Config) !void {
    // Validate port
    if (config.port == 0) {
        std.debug.print("Error: Port must be between 1 and 65535\n", .{});
        return error.InvalidPort;
    }

    // Validate paths exist
    std.fs.cwd().access(config.pretrain_path, .{}) catch |err| {
        std.debug.print("Error: Pretrain dataset not found: {s} ({})\n", .{ config.pretrain_path, err });
        return error.FileNotFound;
    };

    std.fs.cwd().access(config.train_path, .{}) catch |err| {
        std.debug.print("Error: Train dataset not found: {s} ({})\n", .{ config.train_path, err });
        return error.FileNotFound;
    };

    if (config.load_model_path) |path| {
        std.fs.cwd().access(path, .{}) catch |err| {
            std.debug.print("Error: Model file not found: {s} ({})\n", .{ path, err });
            return error.FileNotFound;
        };
    }

    // Validate other parameters
    if (config.max_request_size == 0) {
        std.debug.print("Error: max_request_size must be > 0\n", .{});
        return error.InvalidConfig;
    }

    if (config.max_prompt_length == 0) {
        std.debug.print("Error: max_prompt_length must be > 0\n", .{});
        return error.InvalidConfig;
    }
}

fn handleConnection(state: *ServerState, stream: std.net.Stream, config: *const Config) !void {
    var buffer: [8192]u8 = undefined;
    const bytes_read = stream.read(&buffer) catch |err| {
        std.debug.print("[{}] Error reading from stream: {}\n", .{ std.time.timestamp(), err });
        return err;
    };

    if (bytes_read == 0) return;
    if (bytes_read > config.max_request_size) {
        const err_response = "HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Request too large\"}";
        _ = stream.writeAll(err_response) catch {};
        return;
    }

    const request = buffer[0..bytes_read];

    // Parse HTTP request line
    var lines = std.mem.splitScalar(u8, request, '\n');
    const first_line = lines.next() orelse {
        const err_response = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Malformed request\"}";
        _ = stream.writeAll(err_response) catch {};
        return;
    };

    var parts = std.mem.splitScalar(u8, first_line, ' ');
    const method = parts.next() orelse {
        const err_response = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Malformed request\"}";
        _ = stream.writeAll(err_response) catch {};
        return;
    };
    const path = parts.next() orelse {
        const err_response = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Malformed request\"}";
        _ = stream.writeAll(err_response) catch {};
        return;
    };

    std.debug.print("[{}] {s} {s}\n", .{ std.time.timestamp(), method, path });

    // Handle GET /
    if (std.mem.startsWith(u8, method, "GET") and (std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/index.html"))) {
        const response = try std.fmt.allocPrint(state.allocator, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{s}", .{ index_html.len, index_html });
        defer state.allocator.free(response);
        _ = try stream.writeAll(response);
        return;
    }

    // Handle GET /stats
    if (std.mem.startsWith(u8, method, "GET") and std.mem.eql(u8, path, "/stats")) {
        const vocab_size = state.model.vocab.size();
        const embedding_dim = zigformer.config.embedding_dim;
        const hidden_dim = zigformer.config.hidden_dim;
        const max_seq_len = zigformer.config.max_seq_len;
        const num_heads = zigformer.config.num_heads;

        // Manual JSON construction
        var json = std.ArrayList(u8){};
        defer json.deinit(state.allocator);

        try std.fmt.format(json.writer(state.allocator), "{{\"vocab_size\": {}, \"embedding_dim\": {}, \"hidden_dim\": {}, \"max_seq_len\": {}, \"num_heads\": {}}}", .{ vocab_size, embedding_dim, hidden_dim, max_seq_len, num_heads });

        const response = try std.fmt.allocPrint(state.allocator, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{s}", .{ json.items.len, json.items });
        defer state.allocator.free(response);
        _ = try stream.writeAll(response);
        return;
    }

    // Handle POST /chat
    if (std.mem.startsWith(u8, method, "POST") and std.mem.startsWith(u8, path, "/chat")) {
        // Find body (after \r\n\r\n)
        const body_start = std.mem.indexOf(u8, request, "\r\n\r\n") orelse {
            const err_response = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Missing request body\"}";
            _ = try stream.writeAll(err_response);
            return;
        };
        const body = request[body_start + 4 ..];

        const RequestBody = struct {
            prompt: []const u8,
            top_k: ?usize = null,
            top_p: ?f32 = null,
        };
        const parsed = std.json.parseFromSlice(RequestBody, state.allocator, body, .{ .ignore_unknown_fields = true }) catch {
            const err_response = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Invalid JSON format\"}";
            _ = try stream.writeAll(err_response);
            return;
        };
        defer parsed.deinit();

        const prompt = parsed.value.prompt;
        const top_k = parsed.value.top_k orelse 0;
        const top_p = parsed.value.top_p orelse 0.0;

        // Validate prompt length
        if (prompt.len == 0) {
            const err_response = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Prompt cannot be empty\"}";
            _ = try stream.writeAll(err_response);
            return;
        }

        if (prompt.len > config.max_prompt_length) {
            const err_response = try std.fmt.allocPrint(state.allocator, "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{{\"error\": \"Prompt too long (max {} characters)\"}}", .{config.max_prompt_length});
            defer state.allocator.free(err_response);
            _ = try stream.writeAll(err_response);
            return;
        }

        const formatted_input = try std.fmt.allocPrint(state.allocator, "User: {s}", .{prompt});
        defer state.allocator.free(formatted_input);

        // Run prediction (locked)
        state.mutex.lock();
        const result = blk: {
            if (top_k > 0) {
                break :blk state.model.predictWithSampling(formatted_input, .topk, top_k, 0.0);
            } else if (top_p > 0.0) {
                break :blk state.model.predictWithSampling(formatted_input, .topp, 0, top_p);
            } else {
                break :blk state.model.predict(formatted_input);
            }
        } catch |err| {
            state.mutex.unlock();
            std.debug.print("[{}] Prediction error: {}\n", .{ std.time.timestamp(), err });
            const err_response = "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Model prediction failed\"}";
            _ = try stream.writeAll(err_response);
            return;
        };
        state.mutex.unlock();
        defer state.allocator.free(result);

        // JSON response - manually format to avoid API issues
        // Escape quotes and newlines in result for valid JSON
        // Simple escaping for now (better to use std.json.stringify but we need allocPrint)
        // JSON response - manually format to avoid API issues
        // Simple escaping for quotes and backslashes
        var json_response = std.ArrayList(u8){};
        defer json_response.deinit(state.allocator);
        try json_response.appendSlice(state.allocator, "{\"response\": \"");
        for (result) |c| {
            switch (c) {
                '"' => try json_response.appendSlice(state.allocator, "\\\""),
                '\\' => try json_response.appendSlice(state.allocator, "\\\\"),
                '\n' => try json_response.appendSlice(state.allocator, "\\n"),
                '\r' => try json_response.appendSlice(state.allocator, "\\r"),
                '\t' => try json_response.appendSlice(state.allocator, "\\t"),
                else => try json_response.append(state.allocator, c),
            }
        }
        try json_response.appendSlice(state.allocator, "\"}");

        const response = try std.fmt.allocPrint(state.allocator, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{s}", .{ json_response.items.len, json_response.items });
        defer state.allocator.free(response);
        _ = try stream.writeAll(response);
        return;
    }

    // 404
    const not_found = "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 26\r\n\r\n{\"error\": \"Not found\"}";
    _ = try stream.writeAll(not_found);
}

fn runServer(allocator: std.mem.Allocator, config: Config) !void {
    // 1. Load Model
    std.debug.print("Loading model/vocab...\n", .{});

    var model: *llm.LLM = undefined;

    if (config.load_model_path) |path| {
        std.debug.print("Loading model from {s}...\n", .{path});
        model = try allocator.create(llm.LLM);
        model.* = try llm.LLM.load(allocator, path);
    } else {
        // Only build vocab if we're creating a new model
        var pretrain = try readJsonLines(allocator, config.pretrain_path);
        defer pretrain.deinit();
        var chat = try readJsonLines(allocator, config.train_path);
        defer chat.deinit();

        const v = try buildVocabFromDatasets(allocator, pretrain.value, chat.value);

        std.debug.print("Initializing new model (untrained)...\n", .{});
        model = try allocator.create(llm.LLM);
        model.* = try llm.LLM.init(allocator, v);
    }
    defer {
        model.deinit();
        allocator.destroy(model);
    }

    std.debug.print("Model ready.\n", .{});

    // 2. Start Server
    const address = try std.net.Address.parseIp(config.host, config.port);
    var server = try address.listen(.{ .reuse_address = true });
    defer server.deinit();
    std.debug.print("Listening on http://127.0.0.1:{}\n", .{config.port});

    var state = ServerState{
        .allocator = allocator,
        .model = model,
        .mutex = std.Thread.Mutex{},
    };

    while (true) {
        var conn = try server.accept();
        defer conn.stream.close();

        handleConnection(&state, conn.stream, &config) catch |err| {
            std.debug.print("Error handling connection: {}\n", .{err});
        };
    }
}

fn execGui(ctx: chilli.CommandContext) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Check for config file
    const config_path = try ctx.getFlag("config", []const u8);
    var config = if (config_path.len > 0)
        try loadConfig(allocator, config_path)
    else
        Config{};
    defer config.deinit(allocator);

    // Override with CLI flags if no config file
    if (config_path.len == 0) {
        config.port = @intCast(try ctx.getFlag("port", i64));
        config.host = try ctx.getFlag("host", []const u8);
        config.pretrain_path = try ctx.getFlag("pretrain", []const u8);
        config.train_path = try ctx.getFlag("train", []const u8);

        const load_model = try ctx.getFlag("load-model", []const u8);
        if (load_model.len > 0) config.load_model_path = load_model;
    }

    // Validate configuration
    try validateConfig(config);

    std.debug.print("\n=== ZigFormer GUI Server ===\n", .{});
    std.debug.print("Configuration:\n", .{});
    std.debug.print("  Host: {s}\n", .{config.host});
    std.debug.print("  Port: {}\n", .{config.port});
    std.debug.print("  Max request size: {} bytes\n", .{config.max_request_size});
    std.debug.print("  Max prompt length: {} chars\n", .{config.max_prompt_length});
    std.debug.print("  Timeout: {}s\n\n", .{config.timeout_seconds});

    try runServer(allocator, config);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cmd = try chilli.Command.init(allocator, .{
        .name = "zigformer-gui",
        .description = "Web GUI for ZigFormer",
        .version = "0.1.2",
        .exec = execGui,
    });
    defer cmd.deinit();

    try cmd.addFlag(.{
        .name = "config",
        .description = "Path to JSON configuration file (overrides other flags)",
        .type = .String,
        .default_value = .{ .String = "" },
    });
    try cmd.addFlag(.{
        .name = "host",
        .description = "Host address to bind to",
        .type = .String,
        .default_value = .{ .String = "0.0.0.0" },
    });
    try cmd.addFlag(.{
        .name = "port",
        .description = "Port to listen on",
        .type = .Int,
        .default_value = .{ .Int = 8085 },
    });
    try cmd.addFlag(.{
        .name = "pretrain",
        .description = "Path to pretraining dataset (for vocabulary)",
        .type = .String,
        .default_value = .{ .String = "datasets/simple_dataset/pretrain.json" },
    });
    try cmd.addFlag(.{
        .name = "train",
        .description = "Path to training dataset (for vocabulary)",
        .type = .String,
        .default_value = .{ .String = "datasets/simple_dataset/train.json" },
    });
    try cmd.addFlag(.{
        .name = "load-model",
        .description = "Path to load a pretrained model checkpoint",
        .type = .String,
        .default_value = .{ .String = "" },
    });

    try cmd.run(null);
}

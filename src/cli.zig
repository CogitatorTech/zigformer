const std = @import("std");
const chilli = @import("chilli");
const zigformer = @import("zigformer");
const llm = zigformer.llm;
const vocab = zigformer.vocab;

const JsonData = struct {
    contents: []u8,
    parsed: std.json.Parsed([]const []const u8),

    pub fn deinit(self: *JsonData, allocator: std.mem.Allocator) void {
        self.parsed.deinit();
        allocator.free(self.contents);
    }
};

const Config = struct {
    pretrain_path: []const u8 = "datasets/simple_dataset/pretrain.json",
    train_path: []const u8 = "datasets/simple_dataset/train.json",
    pre_epochs: usize = 10,
    chat_epochs: usize = 10,
    batch_size: usize = 32,
    accumulation_steps: usize = 1,
    pre_lr: f32 = 0.0005,
    chat_lr: f32 = 0.0001,
    save_model_path: ?[]const u8 = null,
    load_model_path: ?[]const u8 = null,
    interactive: bool = true,
    owns_memory: bool = false,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        if (self.owns_memory) {
            allocator.free(self.pretrain_path);
            allocator.free(self.train_path);
            if (self.save_model_path) |p| allocator.free(p);
            if (self.load_model_path) |p| allocator.free(p);
        }
    }
};

fn loadConfig(allocator: std.mem.Allocator, path: []const u8) !Config {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const contents = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(contents);

    const parsed = try std.json.parseFromSlice(Config, allocator, contents, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    // We need to duplicate strings because parsed.deinit() frees them
    var config = parsed.value;
    config.pretrain_path = try allocator.dupe(u8, config.pretrain_path);
    config.train_path = try allocator.dupe(u8, config.train_path);
    if (config.save_model_path) |p| config.save_model_path = try allocator.dupe(u8, p);
    if (config.load_model_path) |p| config.load_model_path = try allocator.dupe(u8, p);
    config.owns_memory = true;

    return config;
}

fn validateConfig(config: Config) void {
    if (config.batch_size == 0) {
        std.debug.print("Error: batch_size must be > 0\n", .{});
        std.process.exit(1);
    }
    if (config.accumulation_steps == 0) {
        std.debug.print("Error: accumulation_steps must be > 0\n", .{});
        std.process.exit(1);
    }
    if (config.pre_lr <= 0) {
        std.debug.print("Error: pre_lr must be > 0\n", .{});
        std.process.exit(1);
    }
    if (config.chat_lr <= 0) {
        std.debug.print("Error: chat_lr must be > 0\n", .{});
        std.process.exit(1);
    }
}
fn splitText(text: []const u8, allocator: std.mem.Allocator) !std.ArrayList([]const u8) {
    var list = std.ArrayList([]const u8){};
    errdefer list.deinit(allocator);
    var it = std.mem.splitScalar(u8, text, ' ');
    while (it.next()) |word| {
        try list.append(allocator, word);
    }
    return list;
}

fn stringLessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

fn readJsonLines(allocator: std.mem.Allocator, path: []const u8) !JsonData {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const contents = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);

    const parsed = std.json.parseFromSlice([]const []const u8, allocator, contents, .{}) catch |err| {
        std.debug.print("Error parsing JSON file: {s}\n", .{path});
        std.debug.print("Ensure it is a valid JSON array of strings.\n", .{});
        allocator.free(contents);
        return err;
    };
    return JsonData{ .contents = contents, .parsed = parsed };
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
    std.sort.pdq([]const u8, vocab_words.items, {}, stringLessThan);

    var v = vocab.Vocab.init(allocator);
    errdefer v.deinit();
    try v.build(vocab_words.items);
    return v;
}

fn parseF32Default(s: []const u8, default_val: f32) f32 {
    if (std.mem.trim(u8, s, " \t\n\r").len == 0) return default_val;
    return std.fmt.parseFloat(f32, s) catch default_val;
}

fn trainAndMaybeRepl(allocator: std.mem.Allocator, pretrain_path: []const u8, chat_path: []const u8, pre_epochs: usize, pre_lr: f32, chat_epochs: usize, chat_lr: f32, batch_size: usize, accumulation_steps: usize, enter_repl: bool, save_model_path: ?[]const u8, load_model_path: ?[]const u8) !void {
    var pretrain = try readJsonLines(allocator, pretrain_path);
    defer pretrain.deinit(allocator);
    var chat = try readJsonLines(allocator, chat_path);
    defer chat.deinit(allocator);

    var v = try buildVocabFromDatasets(allocator, pretrain.parsed.value, chat.parsed.value);
    var v_owned_by_model = false;
    defer if (!v_owned_by_model) v.deinit();

    var model = if (load_model_path) |path| blk: {
        std.debug.print("\n=== LOADING MODEL ===\n", .{});
        std.debug.print("Loading model from: {s}\n", .{path});
        var loaded_model = llm.LLM.load(allocator, path) catch |err| {
            std.debug.print("Error loading model: {}\n", .{err});
            std.debug.print("Initializing new model instead...\n", .{});
            const m = try llm.LLM.init(allocator, v);
            v_owned_by_model = true;
            break :blk m;
        };

        // Check vocab compatibility
        if (loaded_model.vocab.size() != v.size()) {
            std.debug.print("Warning: Loaded model vocab size ({}) doesn't match dataset vocab size ({})\n", .{ loaded_model.vocab.size(), v.size() });
            std.debug.print("This may cause issues. Consider using the same datasets.\n", .{});
        }

        std.debug.print("Model loaded successfully!\n", .{});
        break :blk loaded_model;
    } else blk: {
        const m = try llm.LLM.init(allocator, v);
        v_owned_by_model = true;
        break :blk m;
    };
    defer model.deinit();

    std.debug.print("\n=== MODEL INFORMATION ===\n", .{});
    std.debug.print("Network architecture: {s}\n", .{model.networkDescription()});
    std.debug.print("Model configuration -> max_seq_len: {}, embedding_dim: {}, hidden_dim: {}\n", .{
        zigformer.config.max_seq_len, zigformer.config.embedding_dim, zigformer.config.hidden_dim,
    });
    std.debug.print("Total parameters: {}\n", .{model.totalParameters()});

    const test_string = "User: How do mountains form?";
    std.debug.print("\n=== BEFORE TRAINING ===\n", .{});
    std.debug.print("Input: {s}\n", .{test_string});
    const prediction_before = try model.predict(test_string);
    defer allocator.free(prediction_before);
    std.debug.print("Output: {s}\n", .{prediction_before});

    const pretraining_data = pretrain.parsed.value;
    std.debug.print("\n=== PRE-TRAINING MODEL ===\n", .{});
    std.debug.print("Pre-training on {} examples for {} epochs with learning rate {}\n", .{ pretraining_data.len, pre_epochs, pre_lr });
    try model.train(pretraining_data, pre_epochs, pre_lr, batch_size, accumulation_steps);

    const chat_training_data = chat.parsed.value;
    std.debug.print("\n=== INSTRUCTION TUNING ===\n", .{});
    std.debug.print("Instruction tuning on {} examples for {} epochs with learning rate {}\n", .{ chat_training_data.len, chat_epochs, chat_lr });
    try model.train(chat_training_data, chat_epochs, chat_lr, batch_size, accumulation_steps);

    std.debug.print("\n=== AFTER TRAINING ===\n", .{});
    std.debug.print("Input: {s}\n", .{test_string});
    const prediction_after = try model.predict(test_string);
    defer allocator.free(prediction_after);
    std.debug.print("Output: {s}\n", .{prediction_after});
    std.debug.print("======================\n\n", .{});

    // Save model if path is provided
    if (save_model_path) |path| {
        try model.save(path);
    }

    if (!enter_repl) return;

    std.debug.print("\n--- Interactive Mode ---\n", .{});
    std.debug.print("Type a prompt and press Enter to generate text.\n", .{});
    std.debug.print("Type 'exit' to quit.\n", .{});

    const stdin_file = std.fs.File{ .handle = std.posix.STDIN_FILENO };
    const stdin = stdin_file.deprecatedReader();
    var buffer: [1024]u8 = undefined;
    while (true) {
        std.debug.print("\nEnter prompt: ", .{});
        const input = (try stdin.readUntilDelimiterOrEof(&buffer, '\n')) orelse break;
        if (std.mem.eql(u8, std.mem.trim(u8, input, " \r\n"), "exit")) {
            std.debug.print("Exiting interactive mode.\n", .{});
            break;
        }

        const formatted_input = try std.fmt.allocPrint(allocator, "User: {s}", .{input});
        const result = try model.predict(formatted_input);
        std.debug.print("Model output: {s}\n", .{result});
        allocator.free(formatted_input);
        allocator.free(result);
    }
}

fn execRoot(ctx: chilli.CommandContext) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Check for config file
    const config_path = try ctx.getFlag("config", []const u8);
    var config = if (config_path.len > 0) try loadConfig(allocator, config_path) else Config{};
    defer config.deinit(allocator);

    // Override with CLI flags if provided (checking if they are different from defaults is tricky with chilli,
    // so we'll assume CLI flags take precedence if they are set to non-default values or if we just use them directly.
    // Actually, a better approach is: use config as base, then overwrite with CLI flags.
    // But chilli returns defaults if flag is missing.
    // So we need to know if flag was actually passed. Chilli doesn't easily expose this.
    // For now, let's just use CLI flags if config is NOT present, OR if we want to support overrides,
    // we have to accept that CLI defaults might overwrite config values.
    // To solve this properly:
    // 1. Load config
    // 2. For each field, check if CLI flag was passed (not easy with current chilli usage).
    // Alternative: We only use config if --config is passed, and ignore other flags? No, overrides are good.
    // Let's assume: Config file sets defaults. CLI flags override.
    // But chilli returns default values if flag is missing.
    // So if config has batch_size=64, and CLI default is 32, and user runs without --batch-size, chilli returns 32.
    // If we overwrite config with 32, we lose the config value.
    // We need to check if the flag was present.
    // Since we can't easily do that, let's prioritize CLI flags ONLY if they are explicitly different from our hardcoded defaults?
    // Or simpler: If --config is passed, we use it. We can manually parse args to see if flags are present, but that's messy.

    // Let's stick to the plan: Config file sets values. CLI flags override.
    // If we want CLI to override, we need to know if user typed it.
    // Given the constraints, let's do this:
    // If --config is present, use it.
    // AND we will NOT read other flags if --config is present, to avoid confusion.
    // OR we can say: CLI flags are ignored if --config is present, EXCEPT for interactive/save-model maybe?
    // Let's go with: If --config is present, it is the source of truth.

    if (config_path.len > 0) {
        std.debug.print("Loaded configuration from {s}\n", .{config_path});
    } else {
        // Populate config from CLI flags
        config.pretrain_path = try ctx.getFlag("pretrain", []const u8);
        config.train_path = try ctx.getFlag("train", []const u8);
        config.pre_epochs = @intCast(try ctx.getFlag("pre-epochs", i64));
        config.chat_epochs = @intCast(try ctx.getFlag("chat-epochs", i64));
        config.batch_size = @intCast(try ctx.getFlag("batch-size", i64));
        config.accumulation_steps = @intCast(try ctx.getFlag("accumulation-steps", i64));

        const pre_lr_s = try ctx.getFlag("pre-lr", []const u8);
        const chat_lr_s = try ctx.getFlag("chat-lr", []const u8);
        config.pre_lr = parseF32Default(pre_lr_s, 0.0005);
        config.chat_lr = parseF32Default(chat_lr_s, 0.0001);

        const save_model_path_str = try ctx.getFlag("save-model", []const u8);
        if (save_model_path_str.len > 0) config.save_model_path = save_model_path_str;

        const load_model_path_str = try ctx.getFlag("load-model", []const u8);
        if (load_model_path_str.len > 0) config.load_model_path = load_model_path_str;

        const repl_flag: i64 = try ctx.getFlag("interactive", i64);
        config.interactive = (repl_flag != 0);
    }

    validateConfig(config);

    try trainAndMaybeRepl(allocator, config.pretrain_path, config.train_path, config.pre_epochs, config.pre_lr, config.chat_epochs, config.chat_lr, config.batch_size, config.accumulation_steps, config.interactive, config.save_model_path, config.load_model_path);
}

fn execPredict(ctx: chilli.CommandContext) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const pretrain_path = try ctx.getFlag("pretrain", []const u8);
    const chat_path = try ctx.getFlag("train", []const u8);
    const prompt = try ctx.getFlag("prompt", []const u8);
    const load_model_path = try ctx.getFlag("load-model", []const u8);
    const beam_width: usize = @intCast(try ctx.getFlag("beam-width", i64));

    var model = if (load_model_path.len > 0) blk: {
        break :blk try llm.LLM.load(allocator, load_model_path);
    } else blk: {
        var pretrain = try readJsonLines(allocator, pretrain_path);
        defer pretrain.deinit(allocator);
        var chat = try readJsonLines(allocator, chat_path);
        defer chat.deinit(allocator);

        var v = try buildVocabFromDatasets(allocator, pretrain.parsed.value, chat.parsed.value);
        var vocab_owned_by_model = false;
        defer if (!vocab_owned_by_model) v.deinit();

        const new_model = try llm.LLM.init(allocator, v);
        vocab_owned_by_model = true;
        break :blk new_model;
    };
    defer model.deinit();

    const formatted_input = try std.fmt.allocPrint(allocator, "User: {s}", .{prompt});
    defer allocator.free(formatted_input);

    var result: []u8 = undefined;
    if (beam_width > 1) {
        result = try model.beamSearch(formatted_input, beam_width, 50); // Default 50 max new tokens
    } else {
        result = try model.predict(formatted_input);
    }
    defer allocator.free(result);

    std.debug.print("{s}\n", .{result});
}

pub fn main() anyerror!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var root_cmd = try chilli.Command.init(allocator, .{
        .name = "zigformer-cli",
        .description = "An educational transformer-based LLM in Zig",
        .version = "v0.1.0",
        .exec = execRoot,
    });
    defer root_cmd.deinit();

    try root_cmd.addFlag(.{
        .name = "pretrain",
        .description = "Path to pretraining dataset (JSON array of strings)",
        .type = .String,
        .default_value = .{ .String = "datasets/simple_dataset/pretrain.json" },
    });
    try root_cmd.addFlag(.{
        .name = "train",
        .description = "Path to instruction-tuning dataset (JSON array of strings)",
        .type = .String,
        .default_value = .{ .String = "datasets/simple_dataset/train.json" },
    });
    try root_cmd.addFlag(.{
        .name = "pre-epochs",
        .description = "Number of epochs for pretraining",
        .type = .Int,
        .default_value = .{ .Int = 10 },
    });
    try root_cmd.addFlag(.{
        .name = "config",
        .description = "Path to JSON configuration file (overrides other flags)",
        .type = .String,
        .default_value = .{ .String = "" },
    });
    try root_cmd.addFlag(.{
        .name = "chat-epochs",
        .description = "Number of epochs for instruction tuning",
        .type = .Int,
        .default_value = .{ .Int = 10 },
    });
    try root_cmd.addFlag(.{
        .name = "pre-lr",
        .description = "Learning rate for pretraining (float)",
        .type = .String,
        .default_value = .{ .String = "0.0005" },
    });
    try root_cmd.addFlag(.{
        .name = "chat-lr",
        .description = "Learning rate for instruction tuning (float)",
        .type = .String,
        .default_value = .{ .String = "0.0001" },
    });
    try root_cmd.addFlag(.{
        .name = "interactive",
        .description = "Enter interactive mode after training (1=true, 0=false)",
        .type = .Int,
        .default_value = .{ .Int = 1 },
    });
    try root_cmd.addFlag(.{
        .name = "batch-size",
        .description = "Batch size for training",
        .type = .Int,
        .default_value = .{ .Int = 32 },
    });
    try root_cmd.addFlag(.{
        .name = "accumulation-steps",
        .description = "Number of gradient accumulation steps (effective batch = batch_size × accumulation_steps)",
        .type = .Int,
        .default_value = .{ .Int = 1 },
    });
    try root_cmd.addFlag(.{
        .name = "load-model",
        .description = "Path to load a pre-trained model checkpoint",
        .type = .String,
        .default_value = .{ .String = "" },
    });
    try root_cmd.addFlag(.{
        .name = "save-model",
        .description = "Path to save the trained model",
        .type = .String,
        .default_value = .{ .String = "" },
    });

    var predict_cmd = try chilli.Command.init(allocator, .{
        .name = "predict",
        .description = "Run a single prediction with an untrained model",
        .exec = execPredict,
    });

    try predict_cmd.addFlag(.{
        .name = "prompt",
        .description = "Prompt to run through the model",
        .type = .String,
        .default_value = .{ .String = "" },
    });
    try predict_cmd.addFlag(.{
        .name = "beam-width",
        .description = "Beam width for beam search decoding (default: 1 = greedy)",
        .type = .Int,
        .default_value = .{ .Int = 1 },
    });
    try predict_cmd.addFlag(.{
        .name = "pretrain",
        .description = "Path to pretraining dataset (used to build vocabulary)",
        .type = .String,
        .default_value = .{ .String = "datasets/simple_dataset/pretrain.json" },
    });
    try predict_cmd.addFlag(.{
        .name = "train",
        .description = "Path to training dataset (used to build vocabulary)",
        .type = .String,
        .default_value = .{ .String = "datasets/simple_dataset/train.json" },
    });

    try root_cmd.addSubcommand(predict_cmd);

    try root_cmd.run(null);
}

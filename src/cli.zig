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

fn splitText(text: []const u8, allocator: std.mem.Allocator) !std.ArrayList([]const u8) {
    var list = std.ArrayList([]const u8){};
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

    const parsed = try std.json.parseFromSlice([]const []const u8, allocator, contents, .{});
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

fn trainAndMaybeRepl(allocator: std.mem.Allocator, pretrain_path: []const u8, chat_path: []const u8, pre_epochs: usize, pre_lr: f32, chat_epochs: usize, chat_lr: f32, enter_repl: bool) !void {
    var pretrain = try readJsonLines(allocator, pretrain_path);
    defer pretrain.deinit(allocator);
    var chat = try readJsonLines(allocator, chat_path);
    defer chat.deinit(allocator);

    var v = try buildVocabFromDatasets(allocator, pretrain.parsed.value, chat.parsed.value);
    defer v.deinit();

    var model = try llm.LLM.init(allocator, v);
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
    try model.train(pretraining_data, pre_epochs, pre_lr);

    const chat_training_data = chat.parsed.value;
    std.debug.print("\n=== INSTRUCTION TUNING ===\n", .{});
    std.debug.print("Instruction tuning on {} examples for {} epochs with learning rate {}\n", .{ chat_training_data.len, chat_epochs, chat_lr });
    try model.train(chat_training_data, chat_epochs, chat_lr);

    std.debug.print("\n=== AFTER TRAINING ===\n", .{});
    std.debug.print("Input: {s}\n", .{test_string});
    const prediction_after = try model.predict(test_string);
    defer allocator.free(prediction_after);
    std.debug.print("Output: {s}\n", .{prediction_after});
    std.debug.print("======================\n\n", .{});

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

    const pretrain_path = try ctx.getFlag("pretrain", []const u8);
    const chat_path = try ctx.getFlag("train", []const u8);
    const pre_epochs: usize = @intCast(try ctx.getFlag("pre-epochs", i64));
    const chat_epochs: usize = @intCast(try ctx.getFlag("chat-epochs", i64));
    const pre_lr_s = try ctx.getFlag("pre-lr", []const u8);
    const chat_lr_s = try ctx.getFlag("chat-lr", []const u8);
    const repl_flag: i64 = try ctx.getFlag("interactive", i64);

    const pre_lr = parseF32Default(pre_lr_s, 0.0005);
    const chat_lr = parseF32Default(chat_lr_s, 0.0001);

    try trainAndMaybeRepl(allocator, pretrain_path, chat_path, pre_epochs, pre_lr, chat_epochs, chat_lr, repl_flag != 0);
}

fn execPredict(ctx: chilli.CommandContext) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const pretrain_path = try ctx.getFlag("pretrain", []const u8);
    const chat_path = try ctx.getFlag("train", []const u8);
    const prompt = try ctx.getFlag("prompt", []const u8);

    var pretrain = try readJsonLines(allocator, pretrain_path);
    defer pretrain.deinit(allocator);
    var chat = try readJsonLines(allocator, chat_path);
    defer chat.deinit(allocator);

    var v = try buildVocabFromDatasets(allocator, pretrain.parsed.value, chat.parsed.value);
    defer v.deinit();

    var model = try llm.LLM.init(allocator, v);
    defer model.deinit();

    const formatted_input = try std.fmt.allocPrint(allocator, "User: {s}", .{prompt});
    defer allocator.free(formatted_input);

    const result = try model.predict(formatted_input);
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
        .default_value = .{ .String = "data/pretrain.json" },
    });
    try root_cmd.addFlag(.{
        .name = "train",
        .description = "Path to instruction-tuning dataset (JSON array of strings)",
        .type = .String,
        .default_value = .{ .String = "data/train.json" },
    });
    try root_cmd.addFlag(.{
        .name = "pre-epochs",
        .description = "Number of epochs for pretraining",
        .type = .Int,
        .default_value = .{ .Int = 10 },
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
        .name = "pretrain",
        .description = "Path to pretraining dataset (used to build vocab)",
        .type = .String,
        .default_value = .{ .String = "data/pretrain.json" },
    });
    try predict_cmd.addFlag(.{
        .name = "train",
        .description = "Path to training dataset (used to build vocab)",
        .type = .String,
        .default_value = .{ .String = "data/train.json" },
    });

    try root_cmd.addSubcommand(predict_cmd);

    try root_cmd.run(null);
}

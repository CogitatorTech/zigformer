const std = @import("std");
const zigformer = @import("lib.zig");
const llm = zigformer.llm;
const vocab = zigformer.vocab;

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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const pretrain_path = "data/pretrain.json";
    const chat_path = "data/train.json";

    const pretrain_file = try std.fs.cwd().openFile(pretrain_path, .{});
    defer pretrain_file.close();
    const chat_file = try std.fs.cwd().openFile(chat_path, .{});
    defer chat_file.close();

    const pretrain_contents = try pretrain_file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(pretrain_contents);
    var pretrain_parsed = try std.json.parseFromSlice([]const []const u8, allocator, pretrain_contents, .{});
    defer pretrain_parsed.deinit();

    const chat_contents = try chat_file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(chat_contents);
    var chat_parsed = try std.json.parseFromSlice([]const []const u8, allocator, chat_contents, .{});
    defer chat_parsed.deinit();

    var vocab_set = std.StringHashMap(void).init(allocator);
    defer vocab_set.deinit();

    try vocab_set.put("</s>", {});

    for (pretrain_parsed.value) |text| {
        var words = try splitText(text, allocator);
        defer words.deinit(allocator);
        for (words.items) |word| {
            try vocab_set.put(word, {});
        }
    }

    for (chat_parsed.value) |text| {
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
    defer v.deinit();
    try v.build(vocab_words.items);

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

    const pretraining_data = pretrain_parsed.value;

    std.debug.print("\n=== PRE-TRAINING MODEL ===\n", .{});
    std.debug.print("Pre-training on {} examples for {} epochs with learning rate {}\n", .{ pretraining_data.len, 10, 0.0005 });
    try model.train(pretraining_data, 10, 0.0005);

    const chat_training_data = chat_parsed.value;

    std.debug.print("\n=== INSTRUCTION TUNING ===\n", .{});
    std.debug.print("Instruction tuning on {} examples for {} epochs with learning rate {}\n", .{ chat_training_data.len, 10, 0.0001 });
    try model.train(chat_training_data, 10, 0.0001);

    std.debug.print("\n=== AFTER TRAINING ===\n", .{});
    std.debug.print("Input: {s}\n", .{test_string});
    const prediction_after = try model.predict(test_string);
    defer allocator.free(prediction_after);
    std.debug.print("Output: {s}\n", .{prediction_after});
    std.debug.print("======================\n\n", .{});

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

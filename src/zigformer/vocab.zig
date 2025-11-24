const std = @import("std");

pub const Vocab = struct {
    allocator: std.mem.Allocator,
    encode_map: std.StringHashMap(u32),
    decode_map: std.AutoHashMap(u32, []const u8),
    words: std.ArrayList([]const u8),

    pub fn init(allocator: std.mem.Allocator) Vocab {
        return Vocab{
            .allocator = allocator,
            .encode_map = std.StringHashMap(u32).init(allocator),
            .decode_map = std.AutoHashMap(u32, []const u8).init(allocator),
            .words = std.ArrayList([]const u8){},
        };
    }

    pub fn deinit(self: *Vocab) void {
        self.encode_map.deinit();
        self.decode_map.deinit();
        self.words.deinit(self.allocator);
    }

    pub fn build(self: *Vocab, word_list: []const []const u8) !void {
        for (word_list, 0..) |word, i| {
            const token_id: u32 = @truncate(i);
            try self.encode_map.put(word, token_id);
            try self.decode_map.put(token_id, word);
            try self.words.append(self.allocator, word);
        }
    }

    pub fn encode(self: Vocab, word: []const u8) ?u32 {
        return self.encode_map.get(word);
    }

    pub fn decode(self: Vocab, token_id: u32) ?[]const u8 {
        return self.decode_map.get(token_id);
    }

    pub fn size(self: Vocab) usize {
        return self.words.items.len;
    }

    pub fn save(self: *const Vocab, writer: anytype) !void {
        // Write vocab size
        try writer.writeInt(usize, self.words.items.len, .little);

        // Write each word
        for (self.words.items) |word| {
            try writer.writeInt(usize, word.len, .little);
            try writer.writeAll(word);
        }
    }

    pub fn load(allocator: std.mem.Allocator, reader: anytype) !Vocab {
        var vocab = Vocab.init(allocator);
        errdefer vocab.deinit();

        // Read vocab size
        const vocab_size = try reader.readInt(usize, .little);

        // Read each word
        var word_list = std.ArrayList([]const u8){};
        defer word_list.deinit(allocator);

        for (0..vocab_size) |_| {
            const word_len = try reader.readInt(usize, .little);
            const word = try allocator.alloc(u8, word_len);
            errdefer allocator.free(word);

            try reader.readNoEof(word);
            try word_list.append(allocator, word);
        }

        try vocab.build(word_list.items);
        return vocab;
    }
};

test "Vocab init and build" {
    const allocator = std.testing.allocator;
    var vocab = Vocab.init(allocator);
    defer vocab.deinit();

    const words = &[_][]const u8{ "hello", "world", "</s>" };
    try vocab.build(words);

    try std.testing.expectEqual(@as(usize, 3), vocab.size());
    try std.testing.expectEqual(@as(?u32, 0), vocab.encode("hello"));
    try std.testing.expectEqualStrings("world", vocab.decode(1).?);
    try std.testing.expect(vocab.encode("zig") == null);
}

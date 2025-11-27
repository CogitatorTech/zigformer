//! Vocabulary management and tokenization.
//!
//! Word-level tokenization mapping between string tokens and integer IDs.
//!
//! The vocabulary maintains bidirectional mappings:
//!  - Word → ID (for encoding text)
//!  - ID → Word (for decoding predictions)

const std = @import("std");

/// Vocabulary mapping between tokens and IDs.
///
/// Maintains a simple word-level vocabulary with bidirectional lookup.
/// Words are stored in insertion order, and IDs are assigned sequentially.
pub const Vocab = struct {
    allocator: std.mem.Allocator,
    encode_map: std.StringHashMap(u32), // Map from word string to token ID
    decode_map: std.AutoHashMap(u32, []const u8), // Map from token ID to word string
    words: std.ArrayList([]const u8), // Array of words indexed by token ID
    owns_words: bool, // true if words were allocated by load(), false if references from build()

    pub fn init(allocator: std.mem.Allocator) Vocab {
        return Vocab{
            .allocator = allocator,
            .encode_map = std.StringHashMap(u32).init(allocator),
            .decode_map = std.AutoHashMap(u32, []const u8).init(allocator),
            .words = std.ArrayList([]const u8){},
            .owns_words = false,
        };
    }

    pub fn deinit(self: *Vocab) void {
        // Free allocated words if we own them
        if (self.owns_words) {
            for (self.words.items) |word| {
                self.allocator.free(word);
            }
        }
        self.encode_map.deinit();
        self.decode_map.deinit();
        self.words.deinit(self.allocator);
    }

    pub fn build(self: *Vocab, word_list: []const []const u8) !void {
        std.debug.assert(self.words.items.len == 0);
        self.owns_words = true;

        var built: usize = 0;
        errdefer {
            for (self.words.items[0..built]) |word| {
                self.allocator.free(word);
            }
            self.words.clearRetainingCapacity();
            self.encode_map.clearRetainingCapacity();
            self.decode_map.clearRetainingCapacity();
            self.owns_words = false;
        }

        for (word_list, 0..) |word, i| {
            const token_id: u32 = @truncate(i);
            const owned_word = try self.allocator.dupe(u8, word);
            var keep_word = false;
            defer if (!keep_word) self.allocator.free(owned_word);

            try self.encode_map.put(owned_word, token_id);
            try self.decode_map.put(token_id, owned_word);
            try self.words.append(self.allocator, owned_word);

            keep_word = true;
            built += 1;
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
        errdefer {
            for (word_list.items) |w| allocator.free(w);
        }

        for (0..vocab_size) |_| {
            const word_len = try reader.readInt(usize, .little);
            const word = try allocator.alloc(u8, word_len);
            errdefer allocator.free(word);

            try reader.readNoEof(word);
            try word_list.append(allocator, word);
        }

        try vocab.build(word_list.items);
        for (word_list.items) |w| allocator.free(w);
        vocab.owns_words = true; // We allocated these words, so we own them
        return vocab;
    }

    /// Split text into raw token strings (words and punctuation).
    /// Caller owns the returned ArrayList and its contents.
    /// This function returns slices of the input `text` to avoid allocation where possible.
    pub fn tokenizeRaw(allocator: std.mem.Allocator, text: []const u8) !std.ArrayList([]const u8) {
        var list = std.ArrayList([]const u8){};
        errdefer list.deinit(allocator);

        var it = std.mem.splitScalar(u8, text, ' ');
        while (it.next()) |word| {
            if (word.len == 0) continue;

            // Special case for end token
            if (std.mem.eql(u8, word, "</s>")) {
                try list.append(allocator, word);
                continue;
            }

            var start: usize = 0;
            for (word, 0..) |c, i| {
                if (std.ascii.isPrint(c) and !std.ascii.isAlphanumeric(c)) {
                    // If we have a word before the punctuation, add it
                    if (i > start) {
                        try list.append(allocator, word[start..i]);
                    }

                    // Add the punctuation as its own token
                    try list.append(allocator, word[i .. i + 1]);

                    start = i + 1;
                }
            }
            // Add remaining part of the word
            if (start < word.len) {
                try list.append(allocator, word[start..]);
            }
        }
        return list;
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

test "Vocab load and save memory safety" {
    const allocator = std.testing.allocator;

    // Create and save a vocab
    var vocab1 = Vocab.init(allocator);
    defer vocab1.deinit();

    const words = &[_][]const u8{ "hello", "world", "</s>" };
    try vocab1.build(words);

    // Save to buffer
    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);
    try vocab1.save(writer);

    // Load from buffer
    var stream = std.io.fixedBufferStream(buffer.items);
    const reader = stream.reader();
    var vocab2 = try Vocab.load(allocator, reader);
    defer vocab2.deinit(); // This should properly free allocated words

    // Verify loaded vocab works
    try std.testing.expectEqual(@as(usize, 3), vocab2.size());
    try std.testing.expectEqual(@as(?u32, 0), vocab2.encode("hello"));
}

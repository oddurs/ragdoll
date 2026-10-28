//! Constant tables for the H.264 decoder (ITU-T H.264 clauses 7, 8 and 9).

/// Zig-zag scan of a 4x4 block: scan index -> raster index (Table 8-13, frame macroblocks).
pub const SCAN4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Zig-zag scan of an 8x8 block: scan index -> raster index (Table 8-14, frame macroblocks).
pub const SCAN8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Default scaling lists in zig-zag order (Table 7-3 and 7-4).
pub const DEFAULT_4X4_INTRA: [u8; 16] = [
    6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42,
];
pub const DEFAULT_4X4_INTER: [u8; 16] = [
    10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34,
];
pub const DEFAULT_8X8_INTRA: [u8; 64] = [
    6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23, 23, 23, 23, 23, 23, 25, 25, 25,
    25, 25, 25, 25, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42,
];
pub const DEFAULT_8X8_INTER: [u8; 64] = [
    9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21, 21, 21, 21, 21, 21, 22, 22, 22,
    22, 22, 22, 22, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35,
];

/// `normAdjust4x4(m, i, j)` values `v[m][0..3]` (8-315).
pub const NORM_ADJUST_4X4: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// `normAdjust8x8(m, i, j)` values `v[m][0..6]` (8-318).
pub const NORM_ADJUST_8X8: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

/// Position class of raster index `i` (row `i / 4`, column `i % 4`) in a 4x4 block for [`NORM_ADJUST_4X4`].
pub const fn norm4_class(i: usize) -> usize {
    let (r, c) = (i >> 2, i & 3);
    if r % 2 == 0 && c % 2 == 0 {
        0
    } else if r % 2 == 1 && c % 2 == 1 {
        1
    } else {
        2
    }
}

/// Position class of raster index `i` in an 8x8 block for [`NORM_ADJUST_8X8`].
pub const fn norm8_class(i: usize) -> usize {
    let (r, c) = (i >> 3, i & 7);
    if r % 4 == 0 && c % 4 == 0 {
        0
    } else if r % 2 == 1 && c % 2 == 1 {
        1
    } else if r % 4 == 2 && c % 4 == 2 {
        2
    } else if (r % 4 == 0 && c % 2 == 1) || (r % 2 == 1 && c % 4 == 0) {
        3
    } else if (r % 4 == 0 && c % 4 == 2) || (r % 4 == 2 && c % 4 == 0) {
        4
    } else {
        5
    }
}

/// `QPc` as a function of `qPi` (Table 8-15).
pub const CHROMA_QP: [u8; 52] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39,
    39, 39,
];

/// Chroma QP for luma `qp_y` and a chroma QP index offset (8.5.8, 8-bit).
pub fn chroma_qp(qp_y: i32, offset: i32) -> u8 {
    CHROMA_QP[(qp_y + offset).clamp(0, 51) as usize]
}

/// Deblocking `alpha'` indexed by `indexA` (Table 8-16).
pub const ALPHA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];

/// Deblocking `beta'` indexed by `indexB` (Table 8-16).
pub const BETA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

/// Deblocking `tC0'` indexed by `indexA` and `bS - 1` (Table 8-17).
pub const TC0: [[u8; 3]; 52] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 2, 3],
    [1, 2, 3],
    [2, 2, 3],
    [2, 2, 4],
    [2, 3, 4],
    [2, 3, 4],
    [3, 3, 5],
    [3, 4, 6],
    [3, 4, 6],
    [4, 5, 7],
    [4, 5, 8],
    [4, 6, 9],
    [5, 7, 10],
    [6, 8, 11],
    [6, 8, 13],
    [7, 10, 14],
    [8, 11, 16],
    [9, 12, 18],
    [10, 13, 20],
    [11, 15, 23],
    [13, 17, 25],
];

/// `coded_block_pattern` for `me(v)` code numbers, `ChromaArrayType` 1 or 2 (Table 9-4): `(intra, inter)`.
pub const CBP_ME_420: [(u8, u8); 48] = [
    (47, 0),
    (31, 16),
    (15, 1),
    (0, 2),
    (23, 4),
    (27, 8),
    (29, 32),
    (30, 3),
    (7, 5),
    (11, 10),
    (13, 12),
    (14, 15),
    (39, 47),
    (43, 7),
    (45, 11),
    (46, 13),
    (16, 14),
    (3, 6),
    (5, 9),
    (10, 31),
    (12, 35),
    (19, 37),
    (21, 42),
    (26, 44),
    (28, 33),
    (35, 34),
    (37, 36),
    (42, 40),
    (44, 39),
    (1, 43),
    (2, 45),
    (4, 46),
    (8, 17),
    (17, 18),
    (18, 20),
    (20, 24),
    (24, 19),
    (6, 21),
    (9, 26),
    (22, 28),
    (25, 23),
    (32, 27),
    (33, 29),
    (34, 30),
    (36, 22),
    (40, 25),
    (38, 38),
    (41, 41),
];

/// `coded_block_pattern` for `me(v)` code numbers, `ChromaArrayType` 0 or 3 (Table 9-4): `(intra, inter)`.
pub const CBP_ME_MONO: [(u8, u8); 16] = [
    (15, 0),
    (0, 1),
    (7, 2),
    (11, 4),
    (13, 8),
    (14, 3),
    (3, 5),
    (5, 10),
    (10, 12),
    (12, 15),
    (1, 7),
    (2, 11),
    (4, 13),
    (8, 14),
    (6, 6),
    (9, 9),
];

/// Prediction list usage of a partition: bit 0 = list 0, bit 1 = list 1.
pub const PRED_L0: u8 = 1;
pub const PRED_L1: u8 = 2;
pub const PRED_BI: u8 = 3;

/// Partition shape of an inter macroblock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    P16x16,
    P16x8,
    P8x16,
    P8x8,
}

/// B-slice macroblock types 1..=21 (Table 7-14): shape and per-partition list usage. Index 0 and 22 are handled separately.
pub const B_MB_TYPES: [(Part, [u8; 2]); 23] = [
    (Part::P16x16, [0, 0]), // B_Direct_16x16
    (Part::P16x16, [PRED_L0, 0]),
    (Part::P16x16, [PRED_L1, 0]),
    (Part::P16x16, [PRED_BI, 0]),
    (Part::P16x8, [PRED_L0, PRED_L0]),
    (Part::P8x16, [PRED_L0, PRED_L0]),
    (Part::P16x8, [PRED_L1, PRED_L1]),
    (Part::P8x16, [PRED_L1, PRED_L1]),
    (Part::P16x8, [PRED_L0, PRED_L1]),
    (Part::P8x16, [PRED_L0, PRED_L1]),
    (Part::P16x8, [PRED_L1, PRED_L0]),
    (Part::P8x16, [PRED_L1, PRED_L0]),
    (Part::P16x8, [PRED_L0, PRED_BI]),
    (Part::P8x16, [PRED_L0, PRED_BI]),
    (Part::P16x8, [PRED_L1, PRED_BI]),
    (Part::P8x16, [PRED_L1, PRED_BI]),
    (Part::P16x8, [PRED_BI, PRED_L0]),
    (Part::P8x16, [PRED_BI, PRED_L0]),
    (Part::P16x8, [PRED_BI, PRED_L1]),
    (Part::P8x16, [PRED_BI, PRED_L1]),
    (Part::P16x8, [PRED_BI, PRED_BI]),
    (Part::P8x16, [PRED_BI, PRED_BI]),
    (Part::P8x8, [0, 0]), // B_8x8
];

/// Sub-macroblock partition: `(count, width_4x4, height_4x4, list usage)`. `count == 0` marks B_Direct_8x8.
pub const P_SUB_TYPES: [(u8, u8, u8, u8); 4] = [
    (1, 2, 2, PRED_L0),
    (2, 2, 1, PRED_L0),
    (2, 1, 2, PRED_L0),
    (4, 1, 1, PRED_L0),
];

pub const B_SUB_TYPES: [(u8, u8, u8, u8); 13] = [
    (0, 2, 2, 0), // B_Direct_8x8
    (1, 2, 2, PRED_L0),
    (1, 2, 2, PRED_L1),
    (1, 2, 2, PRED_BI),
    (2, 2, 1, PRED_L0),
    (2, 1, 2, PRED_L0),
    (2, 2, 1, PRED_L1),
    (2, 1, 2, PRED_L1),
    (2, 2, 1, PRED_BI),
    (2, 1, 2, PRED_BI),
    (4, 1, 1, PRED_L0),
    (4, 1, 1, PRED_L1),
    (4, 1, 1, PRED_BI),
];

/// Position of `luma4x4BlkIdx` in 4x4-block units within a macroblock: `(x, y)` (6.4.3).
pub const BLK4_XY: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
];

/// `luma4x4BlkIdx` at 4x4-block position `[y][x]` within a macroblock.
pub const BLK4_AT: [[usize; 4]; 4] = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]];

/// Field scan of a 4x4 block: scan index -> raster index (Table 8-13, field macroblocks).
pub const FIELD_SCAN4X4: [usize; 16] = [0, 4, 1, 8, 12, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

/// Field scan of an 8x8 block: scan index -> raster index (Table 8-14, field macroblocks).
#[rustfmt::skip]
pub const FIELD_SCAN8X8: [usize; 64] = [
    0, 8, 16, 1, 9, 24, 32, 17, 2, 25, 40, 48, 56, 33, 10, 3,
    18, 41, 49, 57, 26, 11, 4, 19, 34, 42, 50, 58, 27, 12, 5, 20,
    35, 43, 51, 59, 28, 13, 6, 21, 36, 44, 52, 60, 29, 14, 22, 37,
    45, 53, 61, 30, 7, 15, 38, 46, 54, 62, 23, 31, 39, 47, 55, 63,
];

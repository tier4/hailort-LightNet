#include "colormap.hpp"

const unsigned char colormap[] = {
  255,0,0,
  100,60,0,
  0,0,255,
  0,128,128,
  70,0,0,
  0,255,128,
  200,0,0,
  128,0,255,
  0,255,0,
  255,0,255,
};

const unsigned char semseg_colormap[] = {
  0,0,0,
  70,70,70,
  255,0,255,
  30,170,250,
  0,220,220,
  60,20,220,
  142,0,0,
  230,0,0,
  128,64,128,
  128,128,128,
  194,253,147,
  255,206,135,
};


const unsigned char lane_colormap[] = {
  0,0,0,
  0,255,0,
  255,0,0,
  0,128,128,
  0,0,255,
  128,128,128,
};  

const unsigned char jet_colormap[MAX_DISTANCE][3] = {
  {0, 0, 127},
  {0, 0, 137},
  {0, 0, 146},
  {0, 0, 156},
  {0, 0, 166},
  {0, 0, 176},
  {0, 0, 185},
  {0, 0, 195},
  {0, 0, 205},
  {0, 0, 215},
  {0, 0, 224},
  {0, 0, 234},
  {0, 0, 244},
  {0, 0, 254},
  {0, 0, 255},
  {0, 1, 255},
  {0, 9, 255},
  {0, 18, 255},
  {0, 26, 255},
  {0, 35, 255},
  {0, 43, 255},
  {0, 52, 255},
  {0, 61, 255},
  {0, 69, 255},
  {0, 78, 255},
  {0, 86, 255},
  {0, 95, 255},
  {0, 103, 255},
  {0, 112, 255},
  {0, 121, 255},
  {0, 129, 255},
  {0, 138, 255},
  {0, 146, 255},
  {0, 155, 255},
  {0, 163, 255},
  {0, 172, 255},
  {0, 181, 255},
  {0, 189, 255},
  {0, 198, 255},
  {0, 206, 255},
  {0, 215, 255},
  {0, 223, 251},
  {2, 232, 244},
  {9, 241, 237},
  {16, 249, 230},
  {23, 255, 223},
  {30, 255, 216},
  {36, 255, 209},
  {43, 255, 202},
  {50, 255, 195},
  {57, 255, 189},
  {64, 255, 182},
  {71, 255, 175},
  {78, 255, 168},
  {85, 255, 161},
  {92, 255, 154},
  {99, 255, 147},
  {106, 255, 140},
  {113, 255, 133},
  {119, 255, 126},
  {126, 255, 119},
  {133, 255, 113},
  {140, 255, 106},
  {147, 255, 99},
  {154, 255, 92},
  {161, 255, 85},
  {168, 255, 78},
  {175, 255, 71},
  {182, 255, 64},
  {189, 255, 57},
  {195, 255, 50},
  {202, 255, 43},
  {209, 255, 36},
  {216, 255, 30},
  {223, 255, 23},
  {230, 255, 16},
  {237, 255, 9},
  {244, 248, 2},
  {251, 240, 0},
  {255, 232, 0},
  {255, 224, 0},
  {255, 216, 0},
  {255, 208, 0},
  {255, 200, 0},
  {255, 192, 0},
  {255, 184, 0},
  {255, 176, 0},
  {255, 168, 0},
  {255, 161, 0},
  {255, 153, 0},
  {255, 145, 0},
  {255, 137, 0},
  {255, 129, 0},
  {255, 121, 0},
  {255, 113, 0},
  {255, 105, 0},
  {255, 97, 0},
  {255, 89, 0},
  {255, 81, 0},
  {255, 73, 0},
  {255, 65, 0},
  {255, 57, 0},
  {255, 49, 0},
  {255, 41, 0},
  {255, 34, 0},
  {255, 26, 0},
  {254, 18, 0},
  {244, 10, 0},
  {234, 2, 0},
  {224, 0, 0},
  {215, 0, 0},
  {205, 0, 0},
  {195, 0, 0},
  {185, 0, 0},
  {176, 0, 0},
  {166, 0, 0},
  {156, 0, 0},
  {146, 0, 0},
  {137, 0, 0},
  {127, 0, 0}
};

const unsigned char magma_colormap[MAX_DISTANCE][3] = {
  {251, 252, 191},
  {251, 249, 187},
  {252, 245, 183},
  {252, 241, 179},
  {252, 238, 176},
  {252, 234, 172},
  {252, 230, 168},
  {253, 225, 163},
  {253, 221, 159},
  {253, 218, 156},
  {253, 214, 152},
  {253, 210, 149},
  {253, 207, 146},
  {254, 203, 142},
  {254, 198, 137},
  {254, 194, 134},
  {254, 190, 131},
  {254, 187, 128},
  {254, 183, 125},
  {254, 179, 123},
  {254, 174, 118},
  {254, 170, 116},
  {254, 166, 113},
  {253, 162, 111},
  {253, 159, 108},
  {253, 155, 106},
  {253, 151, 104},
  {252, 146, 101},
  {252, 142, 99},
  {251, 138, 98},
  {251, 134, 96},
  {250, 130, 95},
  {250, 127, 94},
  {249, 123, 93},
  {248, 117, 92},
  {247, 113, 91},
  {246, 110, 91},
  {245, 106, 91},
  {243, 103, 91},
  {242, 99, 92},
  {239, 94, 93},
  {238, 91, 94},
  {236, 88, 95},
  {234, 85, 96},
  {231, 82, 98},
  {229, 80, 99},
  {226, 77, 101},
  {222, 74, 103},
  {220, 72, 105},
  {217, 70, 106},
  {214, 68, 108},
  {211, 66, 109},
  {208, 65, 111},
  {203, 62, 113},
  {200, 61, 114},
  {197, 60, 116},
  {194, 58, 117},
  {190, 57, 118},
  {187, 56, 119},
  {184, 55, 120},
  {179, 53, 122},
  {176, 52, 123},
  {172, 51, 123},
  {169, 50, 124},
  {166, 49, 125},
  {163, 48, 126},
  {159, 47, 126},
  {154, 45, 127},
  {151, 44, 127},
  {148, 43, 128},
  {145, 42, 128},
  {141, 41, 128},
  {138, 40, 129},
  {133, 38, 129},
  {130, 37, 129},
  {127, 36, 129},
  {124, 35, 129},
  {121, 34, 129},
  {118, 33, 129},
  {115, 31, 129},
  {110, 30, 129},
  {107, 28, 128},
  {104, 27, 128},
  {101, 26, 128},
  {97, 24, 127},
  {94, 23, 127},
  {90, 21, 126},
  {87, 20, 125},
  {83, 19, 124},
  {80, 18, 123},
  {77, 17, 122},
  {74, 16, 121},
  {71, 15, 119},
  {66, 15, 116},
  {62, 15, 114},
  {59, 15, 111},
  {55, 15, 108},
  {52, 16, 104},
  {48, 16, 101},
  {45, 16, 96},
  {40, 17, 89},
  {37, 17, 85},
  {34, 17, 80},
  {31, 17, 75},
  {28, 16, 70},
  {26, 16, 65},
  {22, 14, 58},
  {20, 13, 53},
  {17, 12, 49},
  {15, 11, 44},
  {13, 10, 40},
  {11, 8, 36},
  {9, 7, 31},
  {6, 5, 25},
  {4, 4, 21},
  {3, 3, 17},
  {2, 2, 13},
  {1, 1, 9},
  {0, 0, 6},
  {0, 0, 3},
};

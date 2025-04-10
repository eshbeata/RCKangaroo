#ifndef RTX5090_SETTINGS_H
#define RTX5090_SETTINGS_H

// Define optimization mode for RTX 5090
#ifndef RTX5090_MODE
#define RTX5090_MODE
#endif

// Define block size and thread counts optimized for RTX 5090
#ifndef RTX5090_BLOCK_SIZE
#define RTX5090_BLOCK_SIZE 64     // Very small block size
#endif

#ifndef RTX5090_PNT_GROUP_CNT
#define RTX5090_PNT_GROUP_CNT 4   // Minimal group count
#endif

#ifndef RTX5090_STEP_CNT
#define RTX5090_STEP_CNT 64       // Reduced step count
#endif

// Define global optimization flag
#ifndef OPTIMIZE_FOR_RTX5090
#define OPTIMIZE_FOR_RTX5090
#endif

// Small MD_LEN for RTX 5090
#ifndef RTX5090_MD_LEN
#define RTX5090_MD_LEN 8
#endif

// Conservative DP table size for RTX 5090
#ifndef RTX5090_DPTABLE_MAX_CNT
#define RTX5090_DPTABLE_MAX_CNT 4
#endif

#endif // RTX5090_SETTINGS_H
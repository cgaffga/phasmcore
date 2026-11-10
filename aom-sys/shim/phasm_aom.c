// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// One still image (8-bit 4:2:0, full range) -> one AV1 temporal unit, set up as libavif sets up libaom
// for a single image: all-intra usage, one frame (reduced still-picture header), a fixed quantizer,
// tune=iq, the colour description in the sequence header. The caller (phasm-core `photo`) wraps the
// OBUs in the AVIF container. Measured setup: docs/design/app/photo-attachments.md.

#include <stdlib.h>
#include <string.h>
#include "aom/aom_encoder.h"
#include "aom/aomcx.h"

int phasm_aom_encode_i420(const uint8_t *y, const uint8_t *u, const uint8_t *v, uint32_t w, uint32_t h,
                          int quantizer, int speed, int threads, uint8_t **out, size_t *out_len) {
    *out = NULL;
    *out_len = 0;
    if (w < 2 || h < 2 || (w & 1) || (h & 1) || quantizer < 0 || quantizer > 63) return -1;
    aom_codec_iface_t *iface = aom_codec_av1_cx();
    aom_codec_enc_cfg_t cfg;
    if (aom_codec_enc_config_default(iface, &cfg, AOM_USAGE_ALL_INTRA) != AOM_CODEC_OK) return -2;
    cfg.g_w = w;
    cfg.g_h = h;
    cfg.g_profile = 0;
    cfg.g_bit_depth = AOM_BITS_8;
    cfg.g_input_bit_depth = 8;
    cfg.g_threads = threads > 0 ? (unsigned)threads : 1;
    cfg.g_limit = 1;
    cfg.rc_end_usage = AOM_Q;
    cfg.rc_min_quantizer = (unsigned)quantizer;
    cfg.rc_max_quantizer = (unsigned)quantizer;
    aom_codec_ctx_t ctx;
    if (aom_codec_enc_init(&ctx, iface, &cfg, 0) != AOM_CODEC_OK) return -3;
    int rc = -4;
    aom_image_t *img = NULL;
    if (aom_codec_control(&ctx, AOME_SET_CPUUSED, speed) != AOM_CODEC_OK) goto done;
    if (aom_codec_control(&ctx, AOME_SET_CQ_LEVEL, quantizer) != AOM_CODEC_OK) goto done;
    if (aom_codec_control(&ctx, AV1E_SET_COLOR_RANGE, AOM_CR_FULL_RANGE) != AOM_CODEC_OK) goto done;
    if (aom_codec_control(&ctx, AV1E_SET_COLOR_PRIMARIES, AOM_CICP_CP_BT_709) != AOM_CODEC_OK) goto done;
    if (aom_codec_control(&ctx, AV1E_SET_TRANSFER_CHARACTERISTICS, AOM_CICP_TC_SRGB) != AOM_CODEC_OK) goto done;
    if (aom_codec_control(&ctx, AV1E_SET_MATRIX_COEFFICIENTS, AOM_CICP_MC_BT_601) != AOM_CODEC_OK) goto done;
    if (threads > 1 && aom_codec_control(&ctx, AV1E_SET_ROW_MT, 1) != AOM_CODEC_OK) goto done;
    if (aom_codec_set_option(&ctx, "tune", "iq") != AOM_CODEC_OK) goto done;

    rc = -5;
    img = aom_img_alloc(NULL, AOM_IMG_FMT_I420, w, h, 16);
    if (!img) goto done;
    img->range = AOM_CR_FULL_RANGE;
    img->cp = AOM_CICP_CP_BT_709;
    img->tc = AOM_CICP_TC_SRGB;
    img->mc = AOM_CICP_MC_BT_601;
    for (uint32_t r = 0; r < h; r++) memcpy(img->planes[0] + (size_t)r * img->stride[0], y + (size_t)r * w, w);
    uint32_t cw = w / 2, ch = h / 2;
    for (uint32_t r = 0; r < ch; r++) {
        memcpy(img->planes[1] + (size_t)r * img->stride[1], u + (size_t)r * cw, cw);
        memcpy(img->planes[2] + (size_t)r * img->stride[2], v + (size_t)r * cw, cw);
    }

    rc = -6;
    size_t cap = 0, len = 0;
    uint8_t *buf = NULL;
    for (int pass = 0; pass < 2; pass++) {
        // pass 0 sends the frame, pass 1 flushes
        if (aom_codec_encode(&ctx, pass == 0 ? img : NULL, 0, 1, 0) != AOM_CODEC_OK) goto fail;
        aom_codec_iter_t iter = NULL;
        const aom_codec_cx_pkt_t *pkt;
        while ((pkt = aom_codec_get_cx_data(&ctx, &iter)) != NULL) {
            if (pkt->kind != AOM_CODEC_CX_FRAME_PKT) continue;
            size_t n = pkt->data.frame.sz;
            if (len + n > cap) {
                size_t ncap = (len + n) * 2;
                uint8_t *nb = realloc(buf, ncap);
                if (!nb) goto fail;
                buf = nb;
                cap = ncap;
            }
            memcpy(buf + len, pkt->data.frame.buf, n);
            len += n;
        }
    }
    if (len == 0) goto fail;
    *out = buf;
    *out_len = len;
    rc = 0;
    goto done;
fail:
    free(buf);
done:
    if (img) aom_img_free(img);
    aom_codec_destroy(&ctx);
    return rc;
}

void phasm_aom_free(uint8_t *p) { free(p); }

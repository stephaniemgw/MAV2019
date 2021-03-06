/** @file
 *  @brief PPRZLink message header for COPILOT_STATUS in class telemetry
 *
 *  
 *  @see http://paparazziuav.org
 */

#ifndef _VAR_MESSAGES_telemetry_COPILOT_STATUS_H_
#define _VAR_MESSAGES_telemetry_COPILOT_STATUS_H_


#include "pprzlink/pprzlink_device.h"
#include "pprzlink/pprzlink_transport.h"
#include "pprzlink/pprzlink_utils.h"
#include "pprzlink/pprzlink_message.h"


#ifdef __cplusplus
extern "C" {
#endif

#if DOWNLINK

#define DL_COPILOT_STATUS 125
#define PPRZ_MSG_ID_COPILOT_STATUS 125

/**
 * Macro that redirect calls to the default version of pprzlink API
 * Used for compatibility between versions.
 */
#define pprzlink_msg_send_COPILOT_STATUS _send_msg(COPILOT_STATUS,PPRZLINK_DEFAULT_VER)

/**
 * Sends a COPILOT_STATUS message (API V2.0 version)
 *
 * @param msg the pprzlink_msg structure for this message
 * @param _timestamp Mission computer seconds since startup
 * @param _used_memory Percentage of used memory (RAM) of the mission computer rounded up to whole percent
 * @param _used_disk Percentage of used disk of the mission computer rounded up to whole percent
 * @param _status Mission computer status
 * @param _error_code Error codes of the mission computer
 */
static inline void pprzlink_msg_v2_send_COPILOT_STATUS(struct pprzlink_msg * msg, float *_timestamp, uint8_t *_used_memory, uint8_t *_used_disk, uint8_t *_status, uint8_t *_error_code) {
#if PPRZLINK_ENABLE_FD
  long _FD = 0; /* can be an address, an index, a file descriptor, ... */
#endif
  const uint8_t size = msg->trans->size_of(msg, /* msg header overhead */4+4+1+1+1+1);
  if (msg->trans->check_available_space(msg, _FD_ADDR, size)) {
    msg->trans->count_bytes(msg, size);
    msg->trans->start_message(msg, _FD, /* msg header overhead */4+4+1+1+1+1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, &(msg->sender_id), 1);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, msg->receiver_id, NULL);
    uint8_t comp_class = (msg->component_id & 0x0F) << 4 | (1 & 0x0F);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, comp_class, NULL);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, DL_COPILOT_STATUS, "COPILOT_STATUS");
    msg->trans->put_bytes(msg, _FD, DL_TYPE_FLOAT, DL_FORMAT_SCALAR, (void *) _timestamp, 4);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, (void *) _used_memory, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, (void *) _used_disk, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, (void *) _status, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, (void *) _error_code, 1);
    msg->trans->end_message(msg, _FD);
  } else
        msg->trans->overrun(msg);
}

// Compatibility with the protocol v1.0 API
#define pprzlink_msg_v1_send_COPILOT_STATUS pprz_msg_send_COPILOT_STATUS
#define DOWNLINK_SEND_COPILOT_STATUS(_trans, _dev, timestamp, used_memory, used_disk, status, error_code) pprz_msg_send_COPILOT_STATUS(&((_trans).trans_tx), &((_dev).device), AC_ID, timestamp, used_memory, used_disk, status, error_code)
/**
 * Sends a COPILOT_STATUS message (API V1.0 version)
 *
 * @param trans A pointer to the transport_tx structure used for sending the message
 * @param dev A pointer to the link_device structure through which the message will be sent
 * @param ac_id The id of the sender of the message
 * @param _timestamp Mission computer seconds since startup
 * @param _used_memory Percentage of used memory (RAM) of the mission computer rounded up to whole percent
 * @param _used_disk Percentage of used disk of the mission computer rounded up to whole percent
 * @param _status Mission computer status
 * @param _error_code Error codes of the mission computer
 */
static inline void pprz_msg_send_COPILOT_STATUS(struct transport_tx *trans, struct link_device *dev, uint8_t ac_id, float *_timestamp, uint8_t *_used_memory, uint8_t *_used_disk, uint8_t *_status, uint8_t *_error_code) {
    struct pprzlink_msg msg;
    msg.trans = trans;
    msg.dev = dev;
    msg.sender_id = ac_id;
    msg.receiver_id = 0;
    msg.component_id = 0;
    pprzlink_msg_v2_send_COPILOT_STATUS(&msg,_timestamp,_used_memory,_used_disk,_status,_error_code);
}


#else // DOWNLINK

#define DOWNLINK_SEND_COPILOT_STATUS(_trans, _dev, timestamp, used_memory, used_disk, status, error_code) {}
static inline void pprz_send_msg_COPILOT_STATUS(struct transport_tx *trans __attribute__((unused)), struct link_device *dev __attribute__((unused)), uint8_t ac_id __attribute__((unused)), float *_timestamp __attribute__((unused)), uint8_t *_used_memory __attribute__((unused)), uint8_t *_used_disk __attribute__((unused)), uint8_t *_status __attribute__((unused)), uint8_t *_error_code __attribute__((unused))) {}

#endif // DOWNLINK


/** Getter for field timestamp in message COPILOT_STATUS
  *
  * @param _payload : a pointer to the COPILOT_STATUS message
  * @return Mission computer seconds since startup
  */
static inline float pprzlink_get_DL_COPILOT_STATUS_timestamp(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_float(_payload, 4);
}


/** Getter for field used_memory in message COPILOT_STATUS
  *
  * @param _payload : a pointer to the COPILOT_STATUS message
  * @return Percentage of used memory (RAM) of the mission computer rounded up to whole percent
  */
static inline uint8_t pprzlink_get_DL_COPILOT_STATUS_used_memory(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_uint8_t(_payload, 8);
}


/** Getter for field used_disk in message COPILOT_STATUS
  *
  * @param _payload : a pointer to the COPILOT_STATUS message
  * @return Percentage of used disk of the mission computer rounded up to whole percent
  */
static inline uint8_t pprzlink_get_DL_COPILOT_STATUS_used_disk(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_uint8_t(_payload, 9);
}


/** Getter for field status in message COPILOT_STATUS
  *
  * @param _payload : a pointer to the COPILOT_STATUS message
  * @return Mission computer status
  */
static inline uint8_t pprzlink_get_DL_COPILOT_STATUS_status(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_uint8_t(_payload, 10);
}


/** Getter for field error_code in message COPILOT_STATUS
  *
  * @param _payload : a pointer to the COPILOT_STATUS message
  * @return Error codes of the mission computer
  */
static inline uint8_t pprzlink_get_DL_COPILOT_STATUS_error_code(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_uint8_t(_payload, 11);
}


/* Compatibility macros */
#define DL_COPILOT_STATUS_timestamp(_payload) pprzlink_get_DL_COPILOT_STATUS_timestamp(_payload)
#define DL_COPILOT_STATUS_used_memory(_payload) pprzlink_get_DL_COPILOT_STATUS_used_memory(_payload)
#define DL_COPILOT_STATUS_used_disk(_payload) pprzlink_get_DL_COPILOT_STATUS_used_disk(_payload)
#define DL_COPILOT_STATUS_status(_payload) pprzlink_get_DL_COPILOT_STATUS_status(_payload)
#define DL_COPILOT_STATUS_error_code(_payload) pprzlink_get_DL_COPILOT_STATUS_error_code(_payload)



#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _VAR_MESSAGES_telemetry_COPILOT_STATUS_H_


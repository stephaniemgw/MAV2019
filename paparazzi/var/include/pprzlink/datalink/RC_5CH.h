/** @file
 *  @brief PPRZLink message header for RC_5CH in class datalink
 *
 *  
 *  @see http://paparazziuav.org
 */

#ifndef _VAR_MESSAGES_datalink_RC_5CH_H_
#define _VAR_MESSAGES_datalink_RC_5CH_H_


#include "pprzlink/pprzlink_device.h"
#include "pprzlink/pprzlink_transport.h"
#include "pprzlink/pprzlink_utils.h"
#include "pprzlink/pprzlink_message.h"


#ifdef __cplusplus
extern "C" {
#endif

#if DOWNLINK

#define DL_RC_5CH 53
#define PPRZ_MSG_ID_RC_5CH 53

/**
 * Macro that redirect calls to the default version of pprzlink API
 * Used for compatibility between versions.
 */
#define pprzlink_msg_send_RC_5CH _send_msg(RC_5CH,PPRZLINK_DEFAULT_VER)

/**
 * Sends a RC_5CH message (API V2.0 version)
 *
 * @param msg the pprzlink_msg structure for this message
 * @param _ac_id 
 * @param _throttle 
 * @param _roll 
 * @param _pitch 
 * @param _yaw 
 * @param _mode 
 * @param _kill 
 */
static inline void pprzlink_msg_v2_send_RC_5CH(struct pprzlink_msg * msg, uint8_t *_ac_id, uint8_t *_throttle, int8_t *_roll, int8_t *_pitch, int8_t *_yaw, int8_t *_mode, int8_t *_kill) {
#if PPRZLINK_ENABLE_FD
  long _FD = 0; /* can be an address, an index, a file descriptor, ... */
#endif
  const uint8_t size = msg->trans->size_of(msg, /* msg header overhead */4+1+1+1+1+1+1+1);
  if (msg->trans->check_available_space(msg, _FD_ADDR, size)) {
    msg->trans->count_bytes(msg, size);
    msg->trans->start_message(msg, _FD, /* msg header overhead */4+1+1+1+1+1+1+1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, &(msg->sender_id), 1);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, msg->receiver_id, NULL);
    uint8_t comp_class = (msg->component_id & 0x0F) << 4 | (2 & 0x0F);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, comp_class, NULL);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, DL_RC_5CH, "RC_5CH");
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, (void *) _ac_id, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, (void *) _throttle, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_INT8, DL_FORMAT_SCALAR, (void *) _roll, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_INT8, DL_FORMAT_SCALAR, (void *) _pitch, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_INT8, DL_FORMAT_SCALAR, (void *) _yaw, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_INT8, DL_FORMAT_SCALAR, (void *) _mode, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_INT8, DL_FORMAT_SCALAR, (void *) _kill, 1);
    msg->trans->end_message(msg, _FD);
  } else
        msg->trans->overrun(msg);
}

// Compatibility with the protocol v1.0 API
#define pprzlink_msg_v1_send_RC_5CH pprz_msg_send_RC_5CH
#define DOWNLINK_SEND_RC_5CH(_trans, _dev, ac_id, throttle, roll, pitch, yaw, mode, kill) pprz_msg_send_RC_5CH(&((_trans).trans_tx), &((_dev).device), AC_ID, ac_id, throttle, roll, pitch, yaw, mode, kill)
/**
 * Sends a RC_5CH message (API V1.0 version)
 *
 * @param trans A pointer to the transport_tx structure used for sending the message
 * @param dev A pointer to the link_device structure through which the message will be sent
 * @param ac_id The id of the sender of the message
 * @param _ac_id 
 * @param _throttle 
 * @param _roll 
 * @param _pitch 
 * @param _yaw 
 * @param _mode 
 * @param _kill 
 */
static inline void pprz_msg_send_RC_5CH(struct transport_tx *trans, struct link_device *dev, uint8_t ac_id, uint8_t *_ac_id, uint8_t *_throttle, int8_t *_roll, int8_t *_pitch, int8_t *_yaw, int8_t *_mode, int8_t *_kill) {
    struct pprzlink_msg msg;
    msg.trans = trans;
    msg.dev = dev;
    msg.sender_id = ac_id;
    msg.receiver_id = 0;
    msg.component_id = 0;
    pprzlink_msg_v2_send_RC_5CH(&msg,_ac_id,_throttle,_roll,_pitch,_yaw,_mode,_kill);
}


#else // DOWNLINK

#define DOWNLINK_SEND_RC_5CH(_trans, _dev, ac_id, throttle, roll, pitch, yaw, mode, kill) {}
static inline void pprz_send_msg_RC_5CH(struct transport_tx *trans __attribute__((unused)), struct link_device *dev __attribute__((unused)), uint8_t ac_id __attribute__((unused)), uint8_t *_ac_id __attribute__((unused)), uint8_t *_throttle __attribute__((unused)), int8_t *_roll __attribute__((unused)), int8_t *_pitch __attribute__((unused)), int8_t *_yaw __attribute__((unused)), int8_t *_mode __attribute__((unused)), int8_t *_kill __attribute__((unused))) {}

#endif // DOWNLINK


/** Getter for field ac_id in message RC_5CH
  *
  * @param _payload : a pointer to the RC_5CH message
  * @return 
  */
static inline uint8_t pprzlink_get_DL_RC_5CH_ac_id(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_uint8_t(_payload, 4);
}


/** Getter for field throttle in message RC_5CH
  *
  * @param _payload : a pointer to the RC_5CH message
  * @return 
  */
static inline uint8_t pprzlink_get_DL_RC_5CH_throttle(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_uint8_t(_payload, 5);
}


/** Getter for field roll in message RC_5CH
  *
  * @param _payload : a pointer to the RC_5CH message
  * @return 
  */
static inline int8_t pprzlink_get_DL_RC_5CH_roll(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_int8_t(_payload, 6);
}


/** Getter for field pitch in message RC_5CH
  *
  * @param _payload : a pointer to the RC_5CH message
  * @return 
  */
static inline int8_t pprzlink_get_DL_RC_5CH_pitch(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_int8_t(_payload, 7);
}


/** Getter for field yaw in message RC_5CH
  *
  * @param _payload : a pointer to the RC_5CH message
  * @return 
  */
static inline int8_t pprzlink_get_DL_RC_5CH_yaw(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_int8_t(_payload, 8);
}


/** Getter for field mode in message RC_5CH
  *
  * @param _payload : a pointer to the RC_5CH message
  * @return 
  */
static inline int8_t pprzlink_get_DL_RC_5CH_mode(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_int8_t(_payload, 9);
}


/** Getter for field kill in message RC_5CH
  *
  * @param _payload : a pointer to the RC_5CH message
  * @return 
  */
static inline int8_t pprzlink_get_DL_RC_5CH_kill(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_int8_t(_payload, 10);
}


/* Compatibility macros */
#define DL_RC_5CH_ac_id(_payload) pprzlink_get_DL_RC_5CH_ac_id(_payload)
#define DL_RC_5CH_throttle(_payload) pprzlink_get_DL_RC_5CH_throttle(_payload)
#define DL_RC_5CH_roll(_payload) pprzlink_get_DL_RC_5CH_roll(_payload)
#define DL_RC_5CH_pitch(_payload) pprzlink_get_DL_RC_5CH_pitch(_payload)
#define DL_RC_5CH_yaw(_payload) pprzlink_get_DL_RC_5CH_yaw(_payload)
#define DL_RC_5CH_mode(_payload) pprzlink_get_DL_RC_5CH_mode(_payload)
#define DL_RC_5CH_kill(_payload) pprzlink_get_DL_RC_5CH_kill(_payload)



#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _VAR_MESSAGES_datalink_RC_5CH_H_


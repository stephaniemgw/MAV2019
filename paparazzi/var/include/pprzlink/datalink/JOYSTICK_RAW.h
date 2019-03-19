/** @file
 *  @brief PPRZLink message header for JOYSTICK_RAW in class datalink
 *
 *  
 *  @see http://paparazziuav.org
 */

#ifndef _VAR_MESSAGES_datalink_JOYSTICK_RAW_H_
#define _VAR_MESSAGES_datalink_JOYSTICK_RAW_H_


#include "pprzlink/pprzlink_device.h"
#include "pprzlink/pprzlink_transport.h"
#include "pprzlink/pprzlink_utils.h"
#include "pprzlink/pprzlink_message.h"


#ifdef __cplusplus
extern "C" {
#endif

#if DOWNLINK

#define DL_JOYSTICK_RAW 11
#define PPRZ_MSG_ID_JOYSTICK_RAW 11

/**
 * Macro that redirect calls to the default version of pprzlink API
 * Used for compatibility between versions.
 */
#define pprzlink_msg_send_JOYSTICK_RAW _send_msg(JOYSTICK_RAW,PPRZLINK_DEFAULT_VER)

/**
 * Sends a JOYSTICK_RAW message (API V2.0 version)
 *
 * @param msg the pprzlink_msg structure for this message
 * @param _ac_id 
 * @param _roll 
 * @param _pitch 
 * @param _throttle 
 */
static inline void pprzlink_msg_v2_send_JOYSTICK_RAW(struct pprzlink_msg * msg, uint8_t *_ac_id, int8_t *_roll, int8_t *_pitch, int8_t *_throttle) {
#if PPRZLINK_ENABLE_FD
  long _FD = 0; /* can be an address, an index, a file descriptor, ... */
#endif
  const uint8_t size = msg->trans->size_of(msg, /* msg header overhead */4+1+1+1+1);
  if (msg->trans->check_available_space(msg, _FD_ADDR, size)) {
    msg->trans->count_bytes(msg, size);
    msg->trans->start_message(msg, _FD, /* msg header overhead */4+1+1+1+1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, &(msg->sender_id), 1);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, msg->receiver_id, NULL);
    uint8_t comp_class = (msg->component_id & 0x0F) << 4 | (2 & 0x0F);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, comp_class, NULL);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, DL_JOYSTICK_RAW, "JOYSTICK_RAW");
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, (void *) _ac_id, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_INT8, DL_FORMAT_SCALAR, (void *) _roll, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_INT8, DL_FORMAT_SCALAR, (void *) _pitch, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_INT8, DL_FORMAT_SCALAR, (void *) _throttle, 1);
    msg->trans->end_message(msg, _FD);
  } else
        msg->trans->overrun(msg);
}

// Compatibility with the protocol v1.0 API
#define pprzlink_msg_v1_send_JOYSTICK_RAW pprz_msg_send_JOYSTICK_RAW
#define DOWNLINK_SEND_JOYSTICK_RAW(_trans, _dev, ac_id, roll, pitch, throttle) pprz_msg_send_JOYSTICK_RAW(&((_trans).trans_tx), &((_dev).device), AC_ID, ac_id, roll, pitch, throttle)
/**
 * Sends a JOYSTICK_RAW message (API V1.0 version)
 *
 * @param trans A pointer to the transport_tx structure used for sending the message
 * @param dev A pointer to the link_device structure through which the message will be sent
 * @param ac_id The id of the sender of the message
 * @param _ac_id 
 * @param _roll 
 * @param _pitch 
 * @param _throttle 
 */
static inline void pprz_msg_send_JOYSTICK_RAW(struct transport_tx *trans, struct link_device *dev, uint8_t ac_id, uint8_t *_ac_id, int8_t *_roll, int8_t *_pitch, int8_t *_throttle) {
    struct pprzlink_msg msg;
    msg.trans = trans;
    msg.dev = dev;
    msg.sender_id = ac_id;
    msg.receiver_id = 0;
    msg.component_id = 0;
    pprzlink_msg_v2_send_JOYSTICK_RAW(&msg,_ac_id,_roll,_pitch,_throttle);
}


#else // DOWNLINK

#define DOWNLINK_SEND_JOYSTICK_RAW(_trans, _dev, ac_id, roll, pitch, throttle) {}
static inline void pprz_send_msg_JOYSTICK_RAW(struct transport_tx *trans __attribute__((unused)), struct link_device *dev __attribute__((unused)), uint8_t ac_id __attribute__((unused)), uint8_t *_ac_id __attribute__((unused)), int8_t *_roll __attribute__((unused)), int8_t *_pitch __attribute__((unused)), int8_t *_throttle __attribute__((unused))) {}

#endif // DOWNLINK


/** Getter for field ac_id in message JOYSTICK_RAW
  *
  * @param _payload : a pointer to the JOYSTICK_RAW message
  * @return 
  */
static inline uint8_t pprzlink_get_DL_JOYSTICK_RAW_ac_id(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_uint8_t(_payload, 4);
}


/** Getter for field roll in message JOYSTICK_RAW
  *
  * @param _payload : a pointer to the JOYSTICK_RAW message
  * @return 
  */
static inline int8_t pprzlink_get_DL_JOYSTICK_RAW_roll(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_int8_t(_payload, 5);
}


/** Getter for field pitch in message JOYSTICK_RAW
  *
  * @param _payload : a pointer to the JOYSTICK_RAW message
  * @return 
  */
static inline int8_t pprzlink_get_DL_JOYSTICK_RAW_pitch(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_int8_t(_payload, 6);
}


/** Getter for field throttle in message JOYSTICK_RAW
  *
  * @param _payload : a pointer to the JOYSTICK_RAW message
  * @return 
  */
static inline int8_t pprzlink_get_DL_JOYSTICK_RAW_throttle(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_int8_t(_payload, 7);
}


/* Compatibility macros */
#define DL_JOYSTICK_RAW_ac_id(_payload) pprzlink_get_DL_JOYSTICK_RAW_ac_id(_payload)
#define DL_JOYSTICK_RAW_roll(_payload) pprzlink_get_DL_JOYSTICK_RAW_roll(_payload)
#define DL_JOYSTICK_RAW_pitch(_payload) pprzlink_get_DL_JOYSTICK_RAW_pitch(_payload)
#define DL_JOYSTICK_RAW_throttle(_payload) pprzlink_get_DL_JOYSTICK_RAW_throttle(_payload)



#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _VAR_MESSAGES_datalink_JOYSTICK_RAW_H_

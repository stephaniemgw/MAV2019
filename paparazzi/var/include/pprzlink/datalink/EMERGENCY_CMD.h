/** @file
 *  @brief PPRZLink message header for EMERGENCY_CMD in class datalink
 *
 *  Overcome setting ID and block ID problems in the case of multiboard autopilots like AP/FBW. With this message a KILL command can be sent to AP and FBW at the same time.
 *  @see http://paparazziuav.org
 */

#ifndef _VAR_MESSAGES_datalink_EMERGENCY_CMD_H_
#define _VAR_MESSAGES_datalink_EMERGENCY_CMD_H_


#include "pprzlink/pprzlink_device.h"
#include "pprzlink/pprzlink_transport.h"
#include "pprzlink/pprzlink_utils.h"
#include "pprzlink/pprzlink_message.h"


#ifdef __cplusplus
extern "C" {
#endif

#if DOWNLINK

#define DL_EMERGENCY_CMD 156
#define PPRZ_MSG_ID_EMERGENCY_CMD 156

/**
 * Macro that redirect calls to the default version of pprzlink API
 * Used for compatibility between versions.
 */
#define pprzlink_msg_send_EMERGENCY_CMD _send_msg(EMERGENCY_CMD,PPRZLINK_DEFAULT_VER)

/**
 * Sends a EMERGENCY_CMD message (API V2.0 version)
 *
 * @param msg the pprzlink_msg structure for this message
 * @param _ac_id 
 * @param _cmd 
 */
static inline void pprzlink_msg_v2_send_EMERGENCY_CMD(struct pprzlink_msg * msg, uint8_t *_ac_id, uint8_t *_cmd) {
#if PPRZLINK_ENABLE_FD
  long _FD = 0; /* can be an address, an index, a file descriptor, ... */
#endif
  const uint8_t size = msg->trans->size_of(msg, /* msg header overhead */4+1+1);
  if (msg->trans->check_available_space(msg, _FD_ADDR, size)) {
    msg->trans->count_bytes(msg, size);
    msg->trans->start_message(msg, _FD, /* msg header overhead */4+1+1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, &(msg->sender_id), 1);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, msg->receiver_id, NULL);
    uint8_t comp_class = (msg->component_id & 0x0F) << 4 | (2 & 0x0F);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, comp_class, NULL);
    msg->trans->put_named_byte(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, DL_EMERGENCY_CMD, "EMERGENCY_CMD");
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, (void *) _ac_id, 1);
    msg->trans->put_bytes(msg, _FD, DL_TYPE_UINT8, DL_FORMAT_SCALAR, (void *) _cmd, 1);
    msg->trans->end_message(msg, _FD);
  } else
        msg->trans->overrun(msg);
}

// Compatibility with the protocol v1.0 API
#define pprzlink_msg_v1_send_EMERGENCY_CMD pprz_msg_send_EMERGENCY_CMD
#define DOWNLINK_SEND_EMERGENCY_CMD(_trans, _dev, ac_id, cmd) pprz_msg_send_EMERGENCY_CMD(&((_trans).trans_tx), &((_dev).device), AC_ID, ac_id, cmd)
/**
 * Sends a EMERGENCY_CMD message (API V1.0 version)
 *
 * @param trans A pointer to the transport_tx structure used for sending the message
 * @param dev A pointer to the link_device structure through which the message will be sent
 * @param ac_id The id of the sender of the message
 * @param _ac_id 
 * @param _cmd 
 */
static inline void pprz_msg_send_EMERGENCY_CMD(struct transport_tx *trans, struct link_device *dev, uint8_t ac_id, uint8_t *_ac_id, uint8_t *_cmd) {
    struct pprzlink_msg msg;
    msg.trans = trans;
    msg.dev = dev;
    msg.sender_id = ac_id;
    msg.receiver_id = 0;
    msg.component_id = 0;
    pprzlink_msg_v2_send_EMERGENCY_CMD(&msg,_ac_id,_cmd);
}


#else // DOWNLINK

#define DOWNLINK_SEND_EMERGENCY_CMD(_trans, _dev, ac_id, cmd) {}
static inline void pprz_send_msg_EMERGENCY_CMD(struct transport_tx *trans __attribute__((unused)), struct link_device *dev __attribute__((unused)), uint8_t ac_id __attribute__((unused)), uint8_t *_ac_id __attribute__((unused)), uint8_t *_cmd __attribute__((unused))) {}

#endif // DOWNLINK


/** Getter for field ac_id in message EMERGENCY_CMD
  *
  * @param _payload : a pointer to the EMERGENCY_CMD message
  * @return 
  */
static inline uint8_t pprzlink_get_DL_EMERGENCY_CMD_ac_id(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_uint8_t(_payload, 4);
}


/** Getter for field cmd in message EMERGENCY_CMD
  *
  * @param _payload : a pointer to the EMERGENCY_CMD message
  * @return 
  */
static inline uint8_t pprzlink_get_DL_EMERGENCY_CMD_cmd(uint8_t * _payload __attribute__((unused)))
{
    return _PPRZ_VAL_uint8_t(_payload, 5);
}


/* Compatibility macros */
#define DL_EMERGENCY_CMD_ac_id(_payload) pprzlink_get_DL_EMERGENCY_CMD_ac_id(_payload)
#define DL_EMERGENCY_CMD_cmd(_payload) pprzlink_get_DL_EMERGENCY_CMD_cmd(_payload)



#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _VAR_MESSAGES_datalink_EMERGENCY_CMD_H_

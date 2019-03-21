/*
 * mav_ai.h
 *
 *  Created on: 21 mrt. 2019
 *      Author: tomhoppenbrouwer
 */

#ifndef SW_AIRBORNE_MODULES_MAV_AI_MAV_AI_H_
#define SW_AIRBORNE_MODULES_MAV_AI_MAV_AI_H_



/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.h"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 */

// settings
extern float oa_color_count_frac;

// functions
extern void mav_ai_init(void);
extern void mav_ai_periodic(void);


#endif /* SW_AIRBORNE_MODULES_MAV_AI_MAV_AI_H_ */

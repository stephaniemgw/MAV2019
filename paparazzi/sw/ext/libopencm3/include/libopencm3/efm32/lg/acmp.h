/*
 * This file is part of the libopencm3 project.
 *
 * Copyright (C) 2015 Kuldeep Singh Dhaka <kuldeepdhaka9@gmail.com>
 *
 * This library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LIBOPENCM3_EFM32_ACMP_H
#define LIBOPENCM3_EFM32_ACMP_H

#include <libopencm3/efm32/memorymap.h>
#include <libopencm3/cm3/common.h>

#define ACMP_CTRL(base)		((base) + 0x000)
#define ACMP_INPUTSEL(base)	((base) + 0x004)
#define ACMP_STATUS(base)	((base) + 0x008)
#define ACMP_IEN(base)		((base) + 0x00C)
#define ACMP_IF(base)		((base) + 0x010)
#define ACMP_IFS(base)		((base) + 0x014)
#define ACMP_IFC(base)		((base) + 0x018)
#define ACMP_ROUTE(base)	((base) + 0x01C)

/* ACMP_CTRL */
#define ACMP_CTRL_FULLBIAS	(1 << 31)
#define ACMP_CTRL_HALFBIAS	(1 << 30)

#define ACMP_CTRL_BIASPROG_SHIFT	(24)
#define ACMP_CTRL_BIASPROG_MASK		(0xF << ACMP_CTRL_BIASPROG_SHIFT)
#define ACMP_CTRL_BIASPROG(v)		\
	(((v) << ACMP_CTRL_BIASPROG_SHIFT) & ACMP_CTRL_BIASPROG_MASK)

#define ACMP_CTRL_IFALL		(1 << 17)
#define ACMP_CTRL_IRISE		(1 << 16)

#define ACMP_CTRL_WARMTIME_SHIFT	(8)
#define ACMP_CTRL_WARMTIME_MASK		(0x7 << ACMP_CTRL_WARMTIME_SHIFT)
#define ACMP_CTRL_WARMTIME(v)		\
	(((v) << ACMP_CTRL_WARMTIME_SHIFT) & ACMP_CTRL_WARMTIME_MASK)
#define ACMP_CTRL_WARMTIME_4CYCLES	ACMP_CTRL_WARMTIME(0)
#define ACMP_CTRL_WARMTIME_8CYCLES	ACMP_CTRL_WARMTIME(1)
#define ACMP_CTRL_WARMTIME_16CYCLES	ACMP_CTRL_WARMTIME(2)
#define ACMP_CTRL_WARMTIME_32CYCLES	ACMP_CTRL_WARMTIME(3)
#define ACMP_CTRL_WARMTIME_64CYCLES	ACMP_CTRL_WARMTIME(4)
#define ACMP_CTRL_WARMTIME_128CYCLES	ACMP_CTRL_WARMTIME(5)
#define ACMP_CTRL_WARMTIME_256CYCLES	ACMP_CTRL_WARMTIME(6)
#define ACMP_CTRL_WARMTIME_512CYCLES	ACMP_CTRL_WARMTIME(7)

#define ACMP_CTRL_HYSTSEL_SHIFT	(8)
#define ACMP_CTRL_HYSTSEL_MASK		(0x7 << ACMP_CTRL_HYSTSEL_SHIFT)
#define ACMP_CTRL_HYSTSEL(v)		\
	(((v) << ACMP_CTRL_HYSTSEL_SHIFT) & ACMP_CTRL_HYSTSEL_MASK)
#define ACMP_CTRL_HYSTSEL_HYSTx(x)	ACMP_CTRL_HYSTSEL_HYST(x)
#define ACMP_CTRL_HYSTSEL_HYST0		ACMP_CTRL_HYSTSEL_HYSTx(0)
#define ACMP_CTRL_HYSTSEL_HYST1		ACMP_CTRL_HYSTSEL_HYSTx(1)
#define ACMP_CTRL_HYSTSEL_HYST2		ACMP_CTRL_HYSTSEL_HYSTx(2)
#define ACMP_CTRL_HYSTSEL_HYST3		ACMP_CTRL_HYSTSEL_HYSTx(3)
#define ACMP_CTRL_HYSTSEL_HYST4		ACMP_CTRL_HYSTSEL_HYSTx(4)
#define ACMP_CTRL_HYSTSEL_HYST5		ACMP_CTRL_HYSTSEL_HYSTx(5)
#define ACMP_CTRL_HYSTSEL_HYST6		ACMP_CTRL_HYSTSEL_HYSTx(6)
#define ACMP_CTRL_HYSTSEL_HYST7		ACMP_CTRL_HYSTSEL_HYSTx(7)

#define ACMP_CTRL_GPIOINV		(1 << 3)
#define ACMP_CTRL_INACTVAL		(1 << 2)
#define ACMP_CTRL_MUXEN			(1 << 1)
#define ACMP_CTRL_EN			(1 << 0)

/* ACMP_INPUTSEL */
#define ACMP_INPUTSEL_CSRESSEL_SHIFT	(28)
#define ACMP_INPUTSEL_CSRESSEL_MASK	(0x3 << ACMP_INPUTSEL_CSRESSEL_SHIFT)
#define ACMP_INPUTSEL_CSRESSEL(v)		\
	(((v) << ACMP_INPUTSEL_CSRESSEL_SHIFT) & ACMP_INPUTSEL_CSRESSEL_MASK)
#define ACMP_INPUTSEL_CSRESSEL_RESx(x)	ACMP_INPUTSEL_CSRESSEL_RES(x)
#define ACMP_INPUTSEL_CSRESSEL_RES0	ACMP_INPUTSEL_CSRESSEL_RESx(0)
#define ACMP_INPUTSEL_CSRESSEL_RES1	ACMP_INPUTSEL_CSRESSEL_RESx(1)
#define ACMP_INPUTSEL_CSRESSEL_RES2	ACMP_INPUTSEL_CSRESSEL_RESx(2)
#define ACMP_INPUTSEL_CSRESSEL_RES3	ACMP_INPUTSEL_CSRESSEL_RESx(3)

#define ACMP_INPUTSEL_CSRESEN		(1 << 24)
#define ACMP_INPUTSEL_LPREF		(1 << 16)

#define ACMP_INPUTSEL_VDDLEVEL_SHIFT	(8)
#define ACMP_INPUTSEL_VDDLEVEL_MASK	(0x3F << ACMP_INPUTSEL_VDDLEVEL_SHIFT)
#define ACMP_INPUTSEL_VDDLEVEL(v)	\
	(((v) << ACMP_INPUTSEL_VDDLEVEL_SHIFT) & ACMP_INPUTSEL_VDDLEVEL_MASK)

#define ACMP_INPUTSEL_NEGSEL_SHIFT	(8)
#define ACMP_INPUTSEL_NEGSEL_MASK	(0x3F << ACMP_INPUTSEL_NEGSEL_SHIFT)
#define ACMP_INPUTSEL_NEGSEL(v)		\
	(((v) << ACMP_INPUTSEL_NEGSEL_SHIFT) & ACMP_INPUTSEL_NEGSEL_MASK)
#define ACMP_INPUTSEL_NEGSEL_CHx(x)	ACMP_INPUTSEL_NEGSEL(x)
#define ACMP_INPUTSEL_NEGSEL_CH0	ACMP_INPUTSEL_NEGSEL_CHx(0)
#define ACMP_INPUTSEL_NEGSEL_CH1	ACMP_INPUTSEL_NEGSEL_CHx(1)
#define ACMP_INPUTSEL_NEGSEL_CH2	ACMP_INPUTSEL_NEGSEL_CHx(2)
#define ACMP_INPUTSEL_NEGSEL_CH3	ACMP_INPUTSEL_NEGSEL_CHx(3)
#define ACMP_INPUTSEL_NEGSEL_CH4	ACMP_INPUTSEL_NEGSEL_CHx(4)
#define ACMP_INPUTSEL_NEGSEL_CH5	ACMP_INPUTSEL_NEGSEL_CHx(5)
#define ACMP_INPUTSEL_NEGSEL_CH6	ACMP_INPUTSEL_NEGSEL_CHx(6)
#define ACMP_INPUTSEL_NEGSEL_CH7	ACMP_INPUTSEL_NEGSEL_CHx(7)
#define ACMP_INPUTSEL_NEGSEL_1V25	ACMP_INPUTSEL_NEGSEL(8)
#define ACMP_INPUTSEL_NEGSEL_2V5	ACMP_INPUTSEL_NEGSEL(9)
#define ACMP_INPUTSEL_NEGSEL_VDD	ACMP_INPUTSEL_NEGSEL(10)
#define ACMP_INPUTSEL_NEGSEL_CAPSENSE	ACMP_INPUTSEL_NEGSEL(11)
#define ACMP_INPUTSEL_NEGSEL_DAC0CH0	ACMP_INPUTSEL_NEGSEL(12)
#define ACMP_INPUTSEL_NEGSEL_DAC0CH1	ACMP_INPUTSEL_NEGSEL(13)

#define ACMP_INPUTSEL_POSSEL_SHIFT	(0)
#define ACMP_INPUTSEL_POSSEL_MASK	(0x7 << ACMP_INPUTSEL_POSSEL_SHIFT)
#define ACMP_INPUTSEL_POSSEL(v)		\
	(((v) << ACMP_INPUTSEL_LPOSSELL_SHIFT) & ACMP_INPUTSEL_LPOSSELL_MASK)
#define ACMP_INPUTSEL_POSSEL_CHx(x)	ACMP_INPUTSEL_POSSEL(x)
#define ACMP_INPUTSEL_POSSEL_CH0	ACMP_INPUTSEL_POSSEL_CHx(0)
#define ACMP_INPUTSEL_POSSEL_CH1	ACMP_INPUTSEL_POSSEL_CHx(1)
#define ACMP_INPUTSEL_POSSEL_CH2	ACMP_INPUTSEL_POSSEL_CHx(2)
#define ACMP_INPUTSEL_POSSEL_CH3	ACMP_INPUTSEL_POSSEL_CHx(3)
#define ACMP_INPUTSEL_POSSEL_CH4	ACMP_INPUTSEL_POSSEL_CHx(4)
#define ACMP_INPUTSEL_POSSEL_CH5	ACMP_INPUTSEL_POSSEL_CHx(5)
#define ACMP_INPUTSEL_POSSEL_CH6	ACMP_INPUTSEL_POSSEL_CHx(6)
#define ACMP_INPUTSEL_POSSEL_CH7	ACMP_INPUTSEL_POSSEL_CHx(7)

/* ACMP_STATUS */
#define ACMP_STATUS_ACMPOUT		(1 << 1)
#define ACMP_STATUS_ACMPACT		(1 << 0)

/* ACMP_IEN */
#define ACMP_IEN_WARMUP			(1 << 1)
#define ACMP_IEN_EDGE			(1 << 0)

/* ACMP_IF */
#define ACMP_IF_WARMUP			(1 << 1)
#define ACMP_IF_EDGE			(1 << 0)

/* ACMP_IFS */
#define ACMP_IFS_WARMUP			(1 << 1)
#define ACMP_IFS_EDGE			(1 << 0)

/* ACMP_IFC */
#define ACMP_IFC_WARMUP			(1 << 1)
#define ACMP_IFC_EDGE			(1 << 0)

/* ACMP_ROUTE */
#define ACMP_ROUTE_LOCATION_SHIFT	(8)
#define ACMP_ROUTE_LOCATION_MASK	(0x7 << ACMP_ROUTE_LOCATION_SHIFT)
#define ACMP_ROUTE_LOCATION(v)		\
	(((v) << ACMP_ROUTE_LOCATION_SHIFT) & ACMP_ROUTE_LOCATION_MASK)
#define ACMP_ROUTE_LOCATION_LOCx(x)	ACMP_ROUTE_LOCATION(x)
#define ACMP_ROUTE_LOCATION_LOC0	ACMP_ROUTE_LOCATIONx(0)
#define ACMP_ROUTE_LOCATION_LOC1	ACMP_ROUTE_LOCATIONx(1)
#define ACMP_ROUTE_LOCATION_LOC2	ACMP_ROUTE_LOCATIONx(2)

#define ACMP_ROUTE_ACMPPEN		(1 << 0)

#define ACMP0			ACMP0_BASE
#define ACMP0_CTRL		ACMP_CTRL(ACMP0)
#define ACMP0_INPUTSEL		ACMP_INPUTSEL(ACMP0)
#define ACMP0_STATUS		ACMP_STATUS(ACMP0)
#define ACMP0_IEN		ACMP_IEN(ACMP0)
#define ACMP0_IF		ACMP_IF(ACMP0)
#define ACMP0_IFS		ACMP_IFS(ACMP0)
#define ACMP0_IFC		ACMP_IFC(ACMP0)
#define ACMP0_ROUTE		ACMP_ROUTE(ACMP0)

#define ACMP1			ACMP1_BASE
#define ACMP1_CTRL		ACMP_CTRL(ACMP1)
#define ACMP1_INPUTSEL		ACMP_INPUTSEL(ACMP1)
#define ACMP1_STATUS		ACMP_STATUS(ACMP1)
#define ACMP1_IEN		ACMP_IEN(ACMP1)
#define ACMP1_IF		ACMP_IF(ACMP1)
#define ACMP1_IFS		ACMP_IFS(ACMP1)
#define ACMP1_IFC		ACMP_IFC(ACMP1)
#define ACMP1_ROUTE		ACMP_ROUTE(ACMP1)

#endif


// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ranger_msgs:msg/SystemState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__SYSTEM_STATE__STRUCT_H_
#define RANGER_MSGS__MSG__DETAIL__SYSTEM_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'VEHICLE_STATE_NORMAL'.
enum
{
  ranger_msgs__msg__SystemState__VEHICLE_STATE_NORMAL = 0
};

/// Constant 'VEHICLE_STATE_ESTOP'.
enum
{
  ranger_msgs__msg__SystemState__VEHICLE_STATE_ESTOP = 1
};

/// Constant 'VEHICLE_STATE_EXCEPTION'.
enum
{
  ranger_msgs__msg__SystemState__VEHICLE_STATE_EXCEPTION = 2
};

/// Constant 'CONTROL_MODE_RC'.
enum
{
  ranger_msgs__msg__SystemState__CONTROL_MODE_RC = 0
};

/// Constant 'CONTROL_MODE_CAN'.
enum
{
  ranger_msgs__msg__SystemState__CONTROL_MODE_CAN = 1
};

/// Constant 'MOTION_MODE_DUAL_ACKERMAN'.
enum
{
  ranger_msgs__msg__SystemState__MOTION_MODE_DUAL_ACKERMAN = 0
};

/// Constant 'MOTION_MODE_PARALLEL'.
enum
{
  ranger_msgs__msg__SystemState__MOTION_MODE_PARALLEL = 1
};

/// Constant 'MOTION_MODE_SPINNING'.
enum
{
  ranger_msgs__msg__SystemState__MOTION_MODE_SPINNING = 2
};

/// Constant 'MOTION_MODE_SIDE_SLIP'.
enum
{
  ranger_msgs__msg__SystemState__MOTION_MODE_SIDE_SLIP = 3
};

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

/// Struct defined in msg/SystemState in the package ranger_msgs.
typedef struct ranger_msgs__msg__SystemState
{
  std_msgs__msg__Header header;
  /// system state
  uint8_t vehicle_state;
  uint8_t control_mode;
  uint16_t error_code;
  double battery_voltage;
  /// motion state
  uint8_t motion_mode;
} ranger_msgs__msg__SystemState;

// Struct for a sequence of ranger_msgs__msg__SystemState.
typedef struct ranger_msgs__msg__SystemState__Sequence
{
  ranger_msgs__msg__SystemState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ranger_msgs__msg__SystemState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RANGER_MSGS__MSG__DETAIL__SYSTEM_STATE__STRUCT_H_
